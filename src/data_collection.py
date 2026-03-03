# /home/Anonymous/TD-MoE/src/data_collection.py

import torch
from tqdm import tqdm
import gc
import torch.nn as nn
from functools import partial

def get_module_by_name(model, module_name):
    """
    根据模块路径字符串获取模块实例。
    """
    parts = module_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def profle_svdllm_low_resource(
    model,
    model_config,
    calib_loader,
    dev,
    selected_layers,
    cluster_type='global',
    # <--- 修改/新增 START: 为 group 模式增加 cluster_info 参数 --->
    cluster_info=None,
    # <--- 修改/新增 END --->
    args=None,
):
    print(f"--- Starting statistics profiling for SVD-LLM (MoE Whitening) ---")
    print(f"Mode: {cluster_type} whitening")

    # --- 1. 缓存模型浅层的激活值 ---
    with torch.no_grad():
        if hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = model.model.decoder.layers
        else:
            raise ValueError("Cannot find 'layers' in the model structure.")
        
        if hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.to(dev)
        if hasattr(model.model, 'norm'):
            model.model.norm.to(dev)
    
        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (args.whitening_nsamples, args.model_seq_len, model.config.hidden_size),
            dtype=dtype,
            device='cpu'
        )
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, inp, **kwargs):
                current_batch_size = inp.shape[0]
                start_idx, end_idx = cache['i'], cache['i'] + current_batch_size
                inps[start_idx:end_idx, :inp.shape[1], :] = inp.cpu()
                if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                    cache['attention_mask'] = torch.cat((cache.get('attention_mask'), kwargs['attention_mask'].cpu()), dim=0) if cache.get('attention_mask') is not None else kwargs['attention_mask'].cpu()
                if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                    cache['position_ids'] = torch.cat((cache.get('position_ids'), kwargs['position_ids'].cpu()), dim=0) if cache.get('position_ids') is not None else kwargs['position_ids'].cpu()
                cache['i'] += current_batch_size
                raise ValueError("Catcher interrupted forward pass intentionally.")

        original_first_layer = layers[0]
        layers[0] = Catcher(original_first_layer)
        for batch in tqdm(calib_loader, desc="Caching initial activations"):
            try:
                model(**{k: v.to(dev) for k, v in batch.items()})
            except ValueError:
                pass
        layers[0] = original_first_layer
        
        model.cpu()
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    d_model = model.config.hidden_size
    
    up_proj_name = model_config.role_map.get('up')
    if not up_proj_name:
        raise ValueError("model_config.role_map is missing the 'up' key.")    
    
    try:
        # 尝试 Mixtral/Phi-3.5 类型的路径
        moe_block_sample = model.model.layers[0].block_sparse_moe
        up_proj_module = getattr(moe_block_sample.experts[0], up_proj_name)
        intermediate_size = up_proj_module.out_features
    except Exception:
        try:
             moe_block_sample = model.model.layers[0].mlp
             up_proj_module = getattr(moe_block_sample.experts[0], up_proj_name)
             intermediate_size = up_proj_module.out_features
        except Exception:
            # 如果都失败，使用配置文件中的默认值
            intermediate_size = getattr(model.config, 'moe_intermediate_size', d_model * 4)
            print(f"Warning: Could not dynamically determine intermediate_size. Falling back to {intermediate_size}.")
    
    
    all_layers_stats = {}
    
    with torch.set_grad_enabled(True):
        for i in tqdm(selected_layers, desc="Processing Layers for Whitening Stats"):
            layer = layers[i].to(dev)
            handles = []
            
            module_path = model_config.moe_layer_pattern.format(i)
            moe_block = get_module_by_name(model, module_path)

            if cluster_type == 'global':
                current_layer_stats = {}
                # moe_block.hidden_cov_accumulator = 0
                
                moe_block.hidden_cov_accumulator = torch.zeros((d_model, d_model), device=dev, dtype=torch.float32)
                moe_block.intermediate_cov_accumulator = torch.zeros((intermediate_size, intermediate_size), device=dev, dtype=torch.float32)
                moe_block.hidden_grad_cov_accumulator = torch.zeros((d_model, d_model), device=dev, dtype=torch.float32)
                moe_block.intermediate_grad_cov_accumulator = torch.zeros((intermediate_size, intermediate_size), device=dev, dtype=torch.float32)
                                
                
                def forward_hook_hidden(module, inp, out):
                    hidden_states = inp[0].detach().float()
                    if hidden_states.dim() == 2: hidden_states = hidden_states.unsqueeze(0)
                    cov_increment = torch.matmul(hidden_states.transpose(1, 2), hidden_states)
                    module.hidden_cov_accumulator += torch.sum(cov_increment, dim=0)
                handles.append(moe_block.register_forward_hook(forward_hook_hidden))
                
                # moe_block.intermediate_cov_accumulator = 0
                def forward_hook_intermediate(module, inp, out):
                    hidden_states = inp[0].detach().float()
                    if hidden_states.dim() == 2: hidden_states = hidden_states.unsqueeze(0)
                    cov_increment = torch.matmul(hidden_states.transpose(1, 2), hidden_states)
                    moe_block.intermediate_cov_accumulator += torch.sum(cov_increment, dim=0)
                down_proj_name = model_config.role_map.get('down')
                if hasattr(moe_block, 'experts') and down_proj_name:
                    for expert_module in moe_block.experts:
                        if hasattr(expert_module, down_proj_name):
                            handles.append(getattr(expert_module, down_proj_name).register_forward_hook(forward_hook_intermediate))
                
                # moe_block.hidden_grad_cov_accumulator = 0
                # moe_block.intermediate_grad_cov_accumulator = 0
                def backward_hook_hidden(module, grad_input, grad_output):
                    grad = grad_output[0].detach().float()
                    if grad.dim() == 2: grad = grad.unsqueeze(0)
                    cov_increment = torch.matmul(grad.transpose(1, 2), grad)
                    module.hidden_grad_cov_accumulator += torch.sum(cov_increment, dim=0)
                handles.append(moe_block.register_full_backward_hook(backward_hook_hidden))
                
                def backward_hook_intermediate(module, grad_input, grad_output):
                    grad = grad_input[0].detach().float()
                    if grad.dim() == 2: grad = grad.unsqueeze(0)
                    cov_increment = torch.matmul(grad.transpose(1, 2), grad)
                    moe_block.intermediate_grad_cov_accumulator += torch.sum(cov_increment, dim=0)
                if hasattr(moe_block, 'experts') and down_proj_name:
                    for expert_module in moe_block.experts:
                        if hasattr(expert_module, down_proj_name):
                            handles.append(getattr(expert_module, down_proj_name).register_full_backward_hook(backward_hook_intermediate))
            
            elif cluster_type == 'group':
                if cluster_info is None or f"layer_{i}" not in cluster_info:
                    raise ValueError(f"Group mode requires cluster_info for layer {i}, but none was provided.")
                layer_cluster_map = cluster_info[f"layer_{i}"]
                group_names = list(layer_cluster_map.keys())
                group_accumulators = {
                    group: {'hidden_activation_cov': 0, 'intermediate_activation_cov': 0, 'hidden_gradient_cov': 0, 'intermediate_gradient_cov': 0}
                    for group in group_names
                }
                forward_context = {}

                def forward_hook_moe(module, inp, out):
                    hidden_states_3d = inp[0].detach()
                    if hidden_states_3d.dim() == 2: hidden_states_3d = hidden_states_3d.unsqueeze(0)
                    hidden_states_flat = hidden_states_3d.flatten(0, 1)
                    router_logits = out[1].detach()
                    if router_logits.dim() != 2: router_logits = router_logits.flatten(0, -2)
                    _, selected_experts = torch.topk(router_logits, model.config.num_experts_per_tok, dim=-1)
                    forward_context['selected_experts'] = selected_experts
                    for group_name, expert_ids in layer_cluster_map.items():
                        expert_mask = torch.isin(selected_experts, torch.tensor(expert_ids, device=selected_experts.device))
                        token_mask = expert_mask.any(dim=1)
                        if token_mask.any():
                            tokens_for_group = hidden_states_flat[token_mask].float()
                            group_accumulators[group_name]['hidden_activation_cov'] += tokens_for_group.T @ tokens_for_group

                def backward_hook_moe(module, grad_input, grad_output):
                    grad_3d = grad_output[0].detach()
                    if grad_3d.dim() == 2: grad_3d = grad_3d.unsqueeze(0)
                    grad_flat = grad_3d.flatten(0, 1)
                    selected_experts = forward_context.get('selected_experts')
                    if selected_experts is None: return
                    for group_name, expert_ids in layer_cluster_map.items():
                        expert_mask = torch.isin(selected_experts, torch.tensor(expert_ids, device=selected_experts.device))
                        token_mask = expert_mask.any(dim=1)
                        if token_mask.any():
                            grads_for_group = grad_flat[token_mask].float()
                            group_accumulators[group_name]['hidden_gradient_cov'] += grads_for_group.T @ grads_for_group
                
                handles.append(moe_block.register_forward_hook(forward_hook_moe))
                handles.append(moe_block.register_full_backward_hook(backward_hook_moe))

                down_proj_name = model_config.role_map.get('down')
                if not down_proj_name:
                    raise ValueError(f"模型 '{args.model_path}' 在 config.py 的 role_map 中缺少 'down' 键的配置!")
                
                
                if hasattr(moe_block, 'experts'):
                    for expert_module in moe_block.experts:
                        if hasattr(expert_module, down_proj_name):
                            # 只有找到了才会附加钩子
                            handles.append(getattr(expert_module, down_proj_name).register_forward_hook(forward_hook_intermediate))
                        else:
                            # 如果找不到，就直接报错，防止计算错误数据
                            raise AttributeError(f"在层 {i} 的专家模块中未找到名为 '{down_proj_name}' 的子模块。请检查 config.py。")
                                                        
                for group_name, expert_ids in layer_cluster_map.items():
                    for expert_idx in expert_ids:
                        down_proj_module = getattr(moe_block.experts[expert_idx], down_proj_name)
                        def forward_hook_down(group, module, inp, out):
                            intermediate_states = inp[0].detach().float()
                            if intermediate_states.dim() == 3: intermediate_states = intermediate_states.flatten(0,1)
                            group_accumulators[group]['intermediate_activation_cov'] += intermediate_states.T @ intermediate_states
                        def backward_hook_down(group, module, grad_input, grad_output):
                            grad = grad_input[0].detach().float()
                            if grad.dim() == 3: grad = grad.flatten(0,1)
                            group_accumulators[group]['intermediate_gradient_cov'] += grad.T @ grad
                        handles.append(down_proj_module.register_forward_hook(partial(forward_hook_down, group_name)))
                        handles.append(down_proj_module.register_full_backward_hook(partial(backward_hook_down, group_name)))
            else:
                raise ValueError(f"Unknown cluster_type: {cluster_type}")

            num_cached_samples = cache['i']
            for j in tqdm(range(num_cached_samples), desc=f"Layer {i} Samples", leave=False):
                current_inp_gpu = inps[j].unsqueeze(0).to(dev)
                layer_kwargs = {"attention_mask": attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None}
                layer_kwargs = {k: v for k, v in layer_kwargs.items() if v is not None}
                output = layer(current_inp_gpu, **layer_kwargs)
                output_tensor = output[0] if isinstance(output, tuple) else output
                dummy_loss_grad = torch.randn_like(output_tensor)
                output_tensor.backward(gradient=dummy_loss_grad)
                outs[j] = output_tensor.detach().cpu()
                                                           
            for h in handles:
                h.remove()
            
            if cluster_type == 'global':
                current_layer_stats['hidden_activation_cov'] = moe_block.hidden_cov_accumulator.cpu()
                current_layer_stats['intermediate_activation_cov'] = moe_block.intermediate_cov_accumulator.cpu()
                del moe_block.hidden_cov_accumulator
                del moe_block.intermediate_cov_accumulator
                current_layer_stats['hidden_gradient_cov'] = moe_block.hidden_grad_cov_accumulator.cpu()
                current_layer_stats['intermediate_gradient_cov'] = moe_block.intermediate_grad_cov_accumulator.cpu()
                del moe_block.hidden_grad_cov_accumulator
                del moe_block.intermediate_grad_cov_accumulator                
                all_layers_stats[i] = current_layer_stats
            
            elif cluster_type == 'group':
                all_layers_stats[i] = group_accumulators
            
            layer.to('cpu')
            inps = outs
            torch.cuda.empty_cache()
            gc.collect()
            
    num_samples_processed = cache['i']
    if num_samples_processed > 0:
        for i in selected_layers:
            if i in all_layers_stats:
                # <--- 修改/新增 START: 统一归一化逻辑 --->
                if cluster_type == 'global':
                    for key in all_layers_stats[i]:
                        if all_layers_stats[i][key] is not None:
                            all_layers_stats[i][key] /= num_samples_processed
                elif cluster_type == 'group':
                    for group_name in all_layers_stats[i]:
                        for key in all_layers_stats[i][group_name]:
                            if all_layers_stats[i][group_name][key] is not None:
                                all_layers_stats[i][group_name][key] /= num_samples_processed

    return all_layers_stats