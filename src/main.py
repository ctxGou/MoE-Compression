# src/main.py
import argparse
import os
import torch
import json
from pathlib import Path
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_calib_train_data, ppl_eval_sharing
import torch.multiprocessing as mp
from transformers import BitsAndBytesConfig
# from bitsandbytes.functional import dequantize
from bitsandbytes.nn import Linear4bit

# 从我们项目中导入
from config import get_model_config, FFN_ROLE_CONFIGS,get_layer_by_name
from data_collection import get_forward_activations_for_sigma2
# from decomposition import calculate_tucker_ranks, wt_moe_compress, reconstruct_from_components
from tucker_decomposition import calculate_tucker_ranks, tucker_decomposition,whiten_tensor

# def setup_model_and_tokenizer(model_path: str):
#     """加载模型和分词器, 自动处理多GPU"""
#     print(f"Loading model from {model_path}...")
#     memory_limit_per_gpu = "35GiB"
#     num_gpus = torch.cuda.device_count()
#     max_memory = {i: memory_limit_per_gpu for i in range(num_gpus)}
#     max_memory["cpu"] = "98GiB"
#     print(f"Detected {num_gpus} GPUs. Generated max_memory map: {max_memory}")
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,     # 或 load_in_8bit=True
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )    
        
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     # device_map="cpu",
    #     max_memory=max_memory,
    #     offload_folder="./offload",
    #     low_cpu_mem_usage=True,
    #     offload_state_dict=True,
    #     # load_in_8bit=True
    #     # bnb_configs
        
    # )
    
    # from accelerate import dispatch_model, infer_auto_device_map
    # from transformers import AutoModelForCausalLM, AutoTokenizer

    # # 1. 先在 CPU 上加载
    # from transformers import BitsAndBytesConfig

    # print(f"Loading model from {model_path}...")

    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     quantization_config=quant_config,
    #     device_map="auto",  # 自动拆分 GPU/CPU
    # )

    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # model.eval()
    # return model, tokenizer





def main(args):
    
    
    # --- 1. 设置和加载 ---
    print("--- Step 1: Setting up and loading model ---")
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    device = next(model.parameters()).device
    model.gradient_checkpointing_enable()
    
    model_config = get_model_config(args.model_path)
    model_hidden_config = model.config.to_dict()
    
    # 适配不同模型可能存在的维度名称差异
    model_hidden_config['hidden_size'] = model_hidden_config.get('hidden_size') or model_hidden_config.get('d_model')
    model_hidden_config['intermediate_size'] = model_hidden_config.get('intermediate_size') or model_hidden_config.get('moe_intermediate_size')

    # --- 2. 准备校准数据 ---
    print("--- Step 2: Preparing calibration data ---")
    calib_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)

    # --- 3. 获取所有层信息 ---
    print(f"--- Step 3: Collecting statistics for layers {args.layers_to_compress} ---")
    target_layer_names = [model_config.moe_layer_pattern.format(idx) for idx in args.layers_to_compress]
    print(f"{target_layer_names=}")

    # moe_layer_name = model_config.moe_layer_pattern.format(layer_idx)
    # print(f"\n{'='*20} Processing MoE Layer: {layer_idx} ({moe_layer_name}) {'='*20}")
    # print(f"{model_config=}")
    
    # --- 4.收集所有需要的统计数据 ---
    cov_dir = Path(args.save_path) / "covariances"
    cov_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name
    
    # layers_str = "_".join(map(str, args.layers_to_compress))
    # Sigma2_filename = f"{model_name}_Sigma2_layers_{layers_str}_wsamples_{args.whitening_nsamples}.pt"


    for layer_idx in args.layers_to_compress:
        
        
        moe_layer_name = model_config.moe_layer_pattern.format(layer_idx)
        print(f"\n{'='*20} Processing MoE Layer: {layer_idx} ({moe_layer_name}) {'='*20}")
        
        sigma2_filename = f"{model_name}_Sigma2_layer_{layer_idx}_wsamples_{args.whitening_nsamples}.pt"
        cache_path = cov_dir / sigma2_filename
        
        if cache_path.exists():
            print(f"Loading cached all_covariances from {cache_path} ...")
            x_covariances = torch.load(cache_path, map_location='cpu')
        else:
            print(f"No cache found. Computing all_covariances, covariances will be saved to {cache_path} ...")
            print(f"## model device{device=}")
            sigma2_start_time = time.time()
            x_covariances = get_forward_activations_for_sigma2(
                model=model,
                tokenizer=tokenizer,
                calib_data=calib_data,
                target_layer_names=target_layer_names,
                model_config=model_config,
                roles_config=FFN_ROLE_CONFIGS,
                whitening_nsamples=args.whitening_nsamples,
                device=device,
                batch_size=4
            )
            sigma2_end_time = time.time()
            print(f"{moe_layer_name} sigma2 consumes: {sigma2_end_time - sigma2_start_time:.4f} senconds")
            
            torch.save(x_covariances, cache_path)
            print(f"Saved all_covariances to {cache_path}")    

        # --- 5. 处理该层内的每个角色 (gate, up, down) ---
                                    
        # up/gate 的权重形状 (n_expert, d_out, d_in)
        d_in_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_in_key']]
        d_out_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_out_key']]
        num_experts = model.config.num_local_experts

        # 计算ranks
        ranks_up = calculate_tucker_ranks(num_experts, d_out_up, d_in_up, args.ratio)
        ranks_down = [ranks_up[0], ranks_up[2], ranks_up[1]]        
        # print(f"Calculated Tucker ranks (up/gate): {ranks_up}, (down): {ranks_down}")                
        
        # 对每个专家矩阵进行白化                                                
        for role, param_name in model_config.role_map.items():
            
            save_dir = Path(args.save_path) / "decomposition_results" / Path(args.model_path).name / args.decomposition_mode / f"layer_{layer_idx}" / role
            save_dir.mkdir(parents=True, exist_ok=True)
            
            whiten_output_flag = "whitened_outputs" if args.whiten_output else "unwhitened_outputs"

            tucker_components_path = save_dir / f"tucker_components_ratio_{args.ratio}_{whiten_output_flag}.pt"
            whitened_data_path = save_dir / f"whitening_matrices_ratio_{args.ratio}_{whiten_output_flag}.pt"

            # --- 检查最终结果，如果存在则全部跳过 ---
            if tucker_components_path.exists():
                print(f"Found existing decomposition for role '{role}' at {save_dir}, skipping decomposition...")
                continue
            
            print(f"\n--- Compressing role: '{role}' (parameter: {param_name}), decomposition results will be saved to {save_dir} ---")
            
            # --- C. 检查/计算中间的白化结果 ---
            if whitened_data_path.exists():
                print(f"Found cached whitened data. Loading from {whitened_data_path}...")
                whitened_data = torch.load(whitened_data_path, map_location='cpu')
                W_whitened = whitened_data['W_whitened']
                Cholesky_matrices = whitened_data['Cholesky_matrices']     
                           
            else:
                print(f"No whitened data cache found. Computing...")
                 
                # --- 获取原始权重 W ---
                if model_config.is_stacked:
                    # full_weight_name = f"{moe_layer_name}.{param_name}.weight"
                    # W = dict(model.named_parameters())[full_weight_name].data
                    
                    # 访问.weight属性会触发解量化
                    full_weight_module_name = f"{moe_layer_name}.{param_name}"
                    weight_module = get_layer_by_name(model, full_weight_module_name)                    
                    W = weight_module.weight.to(torch.bfloat16).clone()


                else:
                    print(f"Stacking weights for role '{role}'...")
                    expert_weights = []
                    for i in range(model.config.num_local_experts):
                        # expert_weight_name = f"{moe_layer_name}.experts.{i}.{param_name}.weight"
                        # expert_weights.append(dict(model.named_parameters())[expert_weight_name].data)

                        expert_module = get_layer_by_name(model, f"{moe_layer_name}.experts.{i}.{param_name}")

                        if isinstance(expert_module, Linear4bit):
                            # Linear4bit 特殊处理，拿 _weight 属性或者 weight.float()
                            # dequantized_weight = expert_module._get_qweight.float()  # _weight 已是 [out_features, in_features]
                            dequantized_weight = expert_module.weight.to(torch.bfloat16)
                        else:
                            dequantized_weight = expert_module.weight.float()
                            
                        
                        expert_weights.append(dequantized_weight)
                        
                        if i == 0:
                            print(f"## DEBUG: Shape of dequantized weight is {dequantized_weight.shape}")                        
                        
                        expert_weights.append(dequantized_weight)
                    
                    
                    # W = torch.stack(expert_weights, dim=0)    
                    W = torch.cat(expert_weights, dim=0)          
                    print(f"DEBUG W.shape after concat: {W.shape}")
                    
                # --- 执行压缩 ---
                Sigma2 = x_covariances[moe_layer_name][role]['Sigma2']
                print(f"{Sigma2.shape=}")
                
                if args.whiten_output:
                    # Sigma3 = y_covariances[moe_layer_name][role]['Sigma3']
                    pass                
                else:
                    Sigma3 = None
                            
                # 矩阵白化
                print(f"DEBUG {W.shape=}")
                whittening_start_time = time.time()
                W_whitened,Cholesky_matrices = whiten_tensor(W, Sigma2, Sigma3, args.whiten_output, eps = 1e-6, dev=device)
                whittening_end_time = time.time()
                print(f"{moe_layer_name}.{role} whittening consumes: {whittening_end_time - whittening_start_time:.4f} senconds")
                print(f"Saving intermediate whitened data to {whitened_data_path}...")
                torch.save({'W_whitened': W_whitened.cpu(),'Cholesky_matrices': Cholesky_matrices}, whitened_data_path)
            
            # 白化矩阵分解             
            print(f"Performing Tucker decomposition...")   
            if role == 'up' or role == 'gate':
                ranks = ranks_up
            elif role == 'down':
                ranks = ranks_down
            else:
                raise ValueError(f"role wrong: {role}")
            
            print(f"Compression ratio: {args.ratio}, Target ranks: {ranks}")
            
            tucker_start_time = time.time()
            core, factors = tucker_decomposition(W_whitened,ranks)
            tucker_end_time = time.time()
            
            print(f"{moe_layer_name}.{role} tucker decomposition consumes: {tucker_end_time - tucker_start_time:.4f} senconds")
            
            print(f"{core=}")
            print(f"{factors=}")                                   
    
            print(f"Saving final Tucker components to {tucker_components_path}...")
            torch.save({'core': core.cpu(),'factors': [f.cpu() for f in factors]}, tucker_components_path)        
                
                
            del W_whitened, core, factors, Cholesky_matrices
            if 'whitened_data' in locals():
                del whitened_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        print(f"Layer {moe_layer_name} compression process finish!")
    

    
    # --- 重建并替换权重 ---
    # for layer_idx in args.layers_to_compress:
        
    #     moe_layer_name = model_config.moe_layer_pattern.format(layer_idx)
        
    #     for role, param_name in model_config.role_map.items():
            
    #         save_dir = Path(args.save_path) / "decomposition_results" / Path(args.model_path).name / args.decomposition_mode / f"layer_{layer_idx}" / role
    #         components_path = save_dir / f"compressed_components_ratio_{args.ratio}.pt"
    #         whitening_path = save_dir / f"whitening_matrices_ratio_{args.ratio}.pt"            
                
    #         if not components_path.exists() or not whitening_path.exists():
    #             raise FileNotFoundError(f"Missing decomposition for {moe_layer_name} - {role}")
                              
    #         print(f"Reconstructing role '{role}' in {moe_layer_name}...")
    #         W_prime = reconstruct_from_components(components_path, whitening_path, device)
            
    #         if model_config.is_stacked:
    #             full_weight_name = f"{moe_layer_name}.{param_name}.weight"
    #             dict(model.named_parameters())[full_weight_name].data.copy_(W_prime)
    #         else:
    #             for i in range(num_experts):
    #                 expert_weight_name = f"{moe_layer_name}.experts.{i}.{param_name}.weight"
    #                 dict(model.named_parameters())[expert_weight_name].data.copy_(W_prime[i])
    #         print("Weight replacement complete.")

    #     # 清理内存
    #     del all_covariances
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # print("Reconstruction process finish!")

    # # --- 4. 评估 ---
    # print(f"\n{'='*20} Starting Final Evaluation {'='*20}")
    # if not args.params_only:
    #     result = ppl_eval_sharing(model, tokenizer, "cpu", 'test1', datasets=[args.dataset], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, params_only=args.params_only)
        
    #     eval_save_dir = Path(args.save_path) / "evaluation_results" / Path(args.model_path).name
    #     eval_save_dir.mkdir(parents=True, exist_ok=True)
    #     eval_save_path = eval_save_dir / f"{args.dataset}_ppl_ratio_{args.ratio}_layers_{'_'.join(map(str, args.layers_to_compress))}.json"
        
    #     with open(eval_save_path, 'w') as f:
    #         json.dump(result, f, indent=4)
    #     print(f"Evaluation results saved to {eval_save_path}")
    # else:
    #     print("Skipping evaluation as --params_only is set.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whitened Tucker-MoE Compression Framework")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MoE model")
    parser.add_argument("--save_path", type=str, default="./output", help="Path to save results")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="Calibration and evaluation dataset")
    parser.add_argument("--whitening_nsamples", type=int, default=128, help="Number of samples for calibration")
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.25, help="Compression ratio for Tucker decomposition")
    parser.add_argument("--decomposition_mode", type=str, default="global_whitening", choices=["global_whitening", "group_whitening"])
    parser.add_argument("--layers_to_compress", type=int, nargs='+', required=True, help="List of MoE layer indices to compress (e.g., 15 17)")
    parser.add_argument("--whiten_output", action="store_true",help="If set, the output will be whitened. If not set, the output whitening will be skipped.")
    parser.add_argument("--params_only", action="store_true", help="If set, only run compression and skip final evaluation.")
    
    args = parser.parse_args()
    print(f"{args=}")
    start_time = time.time()
    
    main(args)
    
    end_time = time.time()
    print(f"Total time consumption: {end_time - start_time:.4f} seconds")