# /home/Anonymous/TD-MoE/src/run_whitening.py

import argparse
import os
import torch
import json
from pathlib import Path
import gc
import time
from transformers import AutoTokenizer, AutoConfig
from utils import load_cpu_model
from config import get_model_config, FFN_ROLE_CONFIGS
from tucker_decomposition import calculate_tucker_ranks, whiten_tensor, hosvd_decomposition,calculate_tucker_ranks_balanced,tucker_decomposition


def get_covariances_for_role(role: str, whiten_type: str, all_stats: dict):
    """根据角色和白化类型，从加载的统计数据中选择合适的协方差矩阵。"""
    Sigma2, Sigma3 = None, None
    if whiten_type in ['input', 'both']:
        Sigma2 = all_stats.get('hidden_activation_cov') if role in ['gate', 'up'] else all_stats.get('intermediate_activation_cov')
    if whiten_type in ['output', 'both']:
        Sigma3 = all_stats.get('intermediate_gradient_cov') if role in ['gate', 'up'] else all_stats.get('hidden_gradient_cov')
    return Sigma2, Sigma3

def allocate_ratios_by_importance(target_ratio, all_layers, important_layers_list, factor):
    """根据手动指定的重要层列表，计算每层的自适应Ratio"""
    important_set = set(important_layers_list)
    unimportant_set = set(all_layers) - important_set
    
    n_imp = len(important_set)
    n_unimp = len(unimportant_set)
    n_total = len(all_layers)

    if n_imp == 0 or n_unimp == 0 or factor == 1.0: # 如果所有层都一样重要或因子为1，则全部使用目标Ratio
        return {layer: target_ratio for layer in all_layers}

    # 我们求解方程组:
    # 1) r_imp = factor * r_unimp
    # 2) (n_imp * r_imp + n_unimp * r_unimp) / n_total = target_ratio
    
    # 求解 r_unimp
    r_unimp = (n_total * target_ratio) / (factor * n_imp + n_unimp)
    
    # 求解 r_imp
    r_imp = factor * r_unimp
    
    # 安全裁剪，防止ratio超出合理范围 (e.g., > 1.0 or < 0.0)
    r_imp = min(max(r_imp, 0.05), 0.95)
    r_unimp = min(max(r_unimp, 0.05), 0.95)

    final_ratios = {}
    for layer in all_layers:
        if layer in important_set:
            final_ratios[layer] = r_imp
        else:
            final_ratios[layer] = r_unimp
            
    return final_ratios


def main(args):
    model_config = get_model_config(args.model_path)
    temp_config = AutoConfig.from_pretrained(args.model_path,trust_remote_code=True)
    model_hidden_config = temp_config.to_dict()
    model_hidden_config['hidden_size'] = model_hidden_config.get('hidden_size') or model_hidden_config.get('d_model')
    model_hidden_config['intermediate_size'] = model_hidden_config.get('intermediate_size') or model_hidden_config.get('moe_intermediate_size')
    del temp_config
    
    model_name_str = Path(args.model_path).name
    
    # <---  group 或 mixed 模式都需要加载聚类信息 --->
    cluster_data = None
    if args.cluster_type in ["group", "mixed"]:
        cluster_file_path = Path(args.save_path) / "cluster_results" / model_name_str / f"kmeans_groups_{args.num_clusters}.json"
        if not cluster_file_path.exists():
            raise FileNotFoundError(f"评估 '{args.cluster_type}' 模式需要聚类文件，但在路径 '{cluster_file_path}' 未找到。请先运行 run_clustering.py。")
        print(f"为 '{args.cluster_type}' 模式加载聚类数据: {cluster_file_path}")
        with open(cluster_file_path, 'r') as f:
            cluster_data = json.load(f).get("cluster_groups")
            if not cluster_data:
                raise ValueError(f"聚类文件 '{cluster_file_path}' 格式错误或为空。")

    # <--- 自适应Ratio计算逻辑 --->
    adaptive_ratios = {}
    
    # 当提供了 important_layers 列表时，激活自适应模式
    save_dir_for_ratios = Path(args.save_path) / "ratio_profiles"
    save_dir_for_ratios.mkdir(parents=True, exist_ok=True)        
    ratio_save_path = save_dir_for_ratios / f"{model_name_str}_ratio_{args.ratio}_factor_{args.importance_factor}.json"
    
    if args.important_layers and args.cluster_type == 'global':
        print("\n--- 启动手动自适应Ratio模式 ---")
        if ratio_save_path.exists():  
            print(f"Ratio结果已存在，{ratio_save_path}")
            with open(ratio_save_path, 'r') as f:
                plan = json.load(f)
                adaptive_ratios = {int(k): v for k, v in plan['adaptive_ratios'].items()}
                
        else:            
            adaptive_ratios = allocate_ratios_by_importance(
                args.ratio, 
                args.layers_to_compress, 
                args.important_layers, 
                args.importance_factor
            )

            # 保存时把 key 转成字符串，确保JSON里一致
            plan = {
                "metadata": {
                    "model_name": model_name_str,
                    "layers_to_compress": args.layers_to_compress,
                    "target_average_ratio": args.ratio,
                    "importance_factor": args.importance_factor,
                    "important_layers": sorted(set(args.important_layers))
                },
                "adaptive_ratios": {str(k): v for k, v in adaptive_ratios.items()}
            }          

            with open(ratio_save_path, 'w') as f:
                json.dump(plan, f, indent=4)            
            print(f"自适应Ratio分配方案已保存至: {ratio_save_path}")
            
        print(f"{adaptive_ratios=}")

    
    for layer_idx in args.layers_to_compress:
        
        # <--- 混合模式的核心逻辑 --->
        # 1. 确定当前层的处理模式
        current_layer_mode = args.cluster_type
                
        
        if args.cluster_type == 'mixed':
            if args.global_layers and layer_idx in args.global_layers:
                current_layer_mode = 'global'
                print(f"\n-- [Mixed Mode] Layer {layer_idx} will be processed as GLOBAL --")
            else:
                current_layer_mode = 'group'
                print(f"\n-- [Mixed Mode] Layer {layer_idx} will be processed as GROUP --")
        
        # 2. 根据当前模式设置正确的路径
        base_save_dir = Path(args.save_path) / "decomposition_results" / model_name_str / current_layer_mode / args.whiten_type
        
        # 如果是自适应模式且是global，增加一个 'adaptive' 子目录
        if args.important_layers and current_layer_mode == 'global':            
            save_root_dir = Path(args.save_path) / "decomposition_results" / model_name_str / current_layer_mode / "adaptive"
            print(f"自适应模式已激活，分解结果将保存至: {save_root_dir}")            
        else:
            save_root_dir = base_save_dir
        save_root_dir.mkdir(parents=True, exist_ok=True)        
        
        # 3. 检查协方差文件是否存在
        if args.whiten_type != 'none':
            
            print(f"\n-- Layer {layer_idx} 需要白化，当前模式: {current_layer_mode} --")
            cov_dir = Path(args.save_path) / "covariances" / current_layer_mode        
            sigma_filename = f"{model_name_str}_SigmaMatrix_layer_{layer_idx}_{current_layer_mode}_wsamples_{args.whitening_nsamples}.pt"
            cache_path = cov_dir / sigma_filename

            if not cache_path.exists():
                raise FileNotFoundError(f"协方差矩阵文件未找到: {cache_path}。请先为 '{current_layer_mode}' 模式运行 run_covariances.py。")
            
            print(f"加载层 {layer_idx} 的协方差数据 ({current_layer_mode} 模式) from: {cache_path}")
            loaded_stats = torch.load(cache_path, map_location='cpu')
        
        # 4. 加载模型权重
        print("加载模型以提取权重...")
        model, _ = load_cpu_model(args.model_path)
        moe_layer_name = model_config.moe_layer_pattern.format(layer_idx)
        
        # 5. 根据当前模式执行不同的处理逻辑
        ## 5.1 group 模式
        if current_layer_mode == "group":
            pass
            # layer_key = f"layer_{layer_idx}"
            # if layer_key not in cluster_data:
            #     print(f"Warning: No clustering info for layer {layer_idx} found. Skipping.")
            #     continue
            
            # for group_name, expert_indices_in_group in cluster_data[layer_key].items():
            #     if not expert_indices_in_group: continue

            #     if group_name not in loaded_stats:
            #         raise ValueError(f"在协方差文件 {cache_path} 中找不到组 '{group_name}' 的统计数据。")
            #     current_group_stats = loaded_stats[group_name]

            #     d_in_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_in_key']]
            #     d_out_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_out_key']]
            #     ranks_up = calculate_tucker_ranks_balanced(len(expert_indices_in_group), d_out_up, d_in_up, args.ratio)
            #     ranks_down = [ranks_up[0], ranks_up[2], ranks_up[1]]
                
            #     for role, param_name in model_config.role_map.items():
            #         save_filename = f"layer_{layer_idx}_{role}_{group_name}_ratio_{args.ratio}_{args.whiten_type}.pt"
            #         final_save_path = save_root_dir / save_filename
            #         if final_save_path.exists():
            #             print(f"已存在分解结果: {final_save_path}, 跳过。")
            #             continue
                    
            #         expert_weights = [dict(model.named_parameters())[f"{moe_layer_name}.experts.{i}.{param_name}.weight"].data.cpu() for i in expert_indices_in_group]
            #         W = torch.stack(expert_weights, dim=0)
            #         Sigma2, Sigma3 = get_covariances_for_role(role, args.whiten_type, current_group_stats)
            #         W_whitened, Cholesky_matrices = whiten_tensor(W, Sigma2, Sigma3, eps=1e-3, dev='cpu')
            #         ranks = ranks_up if role in ['gate', 'up'] else ranks_down
            #         # core, factors = hosvd_decomposition(W_whitened, ranks)
            #         core, factors = tucker_decomposition(W_whitened, ranks, method=args.decomposition_method)
            #         torch.save({'core': core, 'factors': factors, 'Cholesky': Cholesky_matrices}, final_save_path)
            #         print(f"已保存组件至: {final_save_path}")
        
        ## 5.2 global 模式
        elif current_layer_mode == "global":
            num_experts = model_hidden_config.get('num_local_experts')
            print(f"\n-- Processing Layer {layer_idx} with global {num_experts} experts --")
            d_in_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_in_key']]
            d_out_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_out_key']]
            
            # 获取当前层的Ratio
            current_ratio = adaptive_ratios.get(layer_idx, args.ratio)
            if current_ratio != args.ratio:
                print(f"  - Layer {layer_idx} 使用自适应Ratio: {current_ratio:.4f}")
            else:
                print(f"  - Layer {layer_idx} 使用固定Ratio: {current_ratio:.4f}")            
            
            # ranks_up = calculate_tucker_ranks_balanced(num_experts, d_out_up, d_in_up, current_ratio)          
            ranks_up = calculate_tucker_ranks(num_experts, d_out_up, d_in_up, current_ratio)                    
            ranks_down = [ranks_up[0], ranks_up[2], ranks_up[1]]
            print(f"Global ranks: up={ranks_up}, down={ranks_down}")
                                        
            for role, param_name in model_config.role_map.items():
                                
                if args.important_layers:
                    save_filename = f"layer_{layer_idx}_{role}_ratio_adp_{args.ratio}_{args.whiten_type}.pt"
                else:
                    save_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}.pt"                
                                
                final_save_path = save_root_dir / save_filename
                if final_save_path.exists():
                    print(f"Found existing decomposition for layer {layer_idx}, role '{role}' at {final_save_path}, skipping...")
                    continue
                
                print(f"\n--- Compressing layer {layer_idx}, role: '{role}', results will be saved to {final_save_path} ---")
                
                expert_weights = [dict(model.named_parameters())[f"{moe_layer_name}.experts.{i}.{param_name}.weight"].data.cpu() for i in range(num_experts)]
                W = torch.stack(expert_weights, dim=0)
                print(f"Original weight tensor W shape: {W.shape}")
                
                if args.whiten_type != 'none':
                    
                    Sigma2, Sigma3 = get_covariances_for_role(role, args.whiten_type, loaded_stats)
                    if Sigma2 is not None: print(f"Role '{role}': Selected Sigma2 for input whitening.")
                    if Sigma3 is not None: print(f"Role '{role}': Selected Sigma3 for output whitening.")
                    
                    W_whitened,Cholesky_matrices = whiten_tensor(W, Sigma2, Sigma3, eps= 1e-3, dev='cpu')
                    ranks = ranks_up if role in ['gate', 'up'] else ranks_down
                    # core, factors = hosvd_decomposition(W_whitened, ranks)
                    core, factors = tucker_decomposition(W_whitened, ranks, method=args.decomposition_method)
                
                else:
                    ranks = ranks_up if role in ['gate', 'up'] else ranks_down
                    # core, factors = hosvd_decomposition(W, ranks)
                    core, factors = tucker_decomposition(W_whitened, ranks, method=args.decomposition_method)
                    Cholesky_matrices = {'S2': None, 'S3': None}
                
                torch.save({'core': core, 'factors': factors, 'Cholesky': Cholesky_matrices}, final_save_path)
                print(f"Saved final components to {final_save_path}")

                
        print(f"\nFinished processing all roles for layer {layer_idx}. Deleting model from memory...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whitened Tucker-MoE Compression Framework")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--whitening_nsamples", type=int, default=256)
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--cluster_type", type=str, default="global", choices=["global", "group", "mixed"])
    parser.add_argument("--layers_to_compress", type=int, nargs='+', required=True)
    parser.add_argument("--whiten_type", type=str, required=True, choices=["input", "output", "both", "none"])
    parser.add_argument("--global_layers", type=int, nargs='+', default=None, help="List of layer indices to be treated as 'global' in 'mixed' mode.")
    parser.add_argument("--num_clusters", type=int, default=2)
    
    # parser.add_argument("--important_layers", type=int, nargs='+', default=None, help="手动指定的重要性层列表")
    # parser.add_argument("--importance_factor", type=float, default=1.5, help="重要层相对于次要层的Ratio倍数")

    parser.add_argument("--decomposition_method", type=str, default="svd", choices=["svd", "qr", "rand"], help="用于Tucker分解的基底生成方法: svd, qr (QR分解), 或 rand (随机SVD)")
    
    args = parser.parse_args()
    print(f"传入参数: {args}")
    
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.4f} 秒")