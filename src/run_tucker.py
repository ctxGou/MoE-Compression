import argparse
import os
import torch
import json
import pandas as pd
from pathlib import Path
import gc
import time
from transformers import AutoTokenizer, AutoConfig
from utils import load_cpu_model, get_calib_train_data
from config import get_model_config, FFN_ROLE_CONFIGS
from tucker_decomposition import whiten_tensor, hosvd_decomposition
from data_collection import profle_svdllm_low_resource
from tucker_decomposition import (
    calculate_tucker_ranks,
    calculate_tucker_ranks_equal,
    whiten_tensor,
    hosvd_decomposition,
    calculate_tucker_ranks_balanced,
)
from utils import load_cpu_model,load_fp16_model
from tqdm import tqdm
from config import get_model_config, set_layer_by_name, get_layer_by_name
from evaluator import run_lm_eval, ppl_eval_sharing
from tucker_decomposition import tucker_decomposition

import run_evaluation

def _ratio_to_str(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _resolve_effective_local_ratio(args, total_moe_layers: int) -> float:
    if args.ratio_scope == "local":
        return float(args.ratio)

    num_selected = len(args.layers_to_compress or [])
    if num_selected <= 0:
        raise ValueError("`layers_to_compress` 不能为空。")

    effective = float(args.ratio) * float(total_moe_layers) / float(num_selected)
    if not (0.0 < effective < 1.0):
        raise ValueError(
            f"全局压缩目标不可行: ratio={args.ratio}, total_moe_layers={total_moe_layers}, "
            f"selected_layers={num_selected} -> local_ratio={effective:.6f} (应在 (0,1) 内)"
        )
    return effective


def get_covariances_for_role(role: str, whiten_type: str, all_stats: dict):
    """根据角色和白化类型，从加载的统计数据中选择合适的协方差矩阵。"""
    Sigma2, Sigma3 = None, None
    if whiten_type in ['input', 'both']:
        Sigma2 = all_stats.get('hidden_activation_cov') if role in ['gate', 'up'] else all_stats.get('intermediate_activation_cov')
    if whiten_type in ['output', 'both']:
        Sigma3 = all_stats.get('intermediate_gradient_cov') if role in ['gate', 'up'] else all_stats.get('hidden_gradient_cov')
    return Sigma2, Sigma3

def run_covariances(args):
    """
    负责计算并缓存协方差矩阵。
    """
    print("--- 启动协方差计算 ---")
    model, tokenizer = load_cpu_model(args.model_path)
    
    model_config = get_model_config(args.model_path)
    model_name = Path(args.model_path).name

    print("--- 准备校准数据 ---")
    calib_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
    
    cov_dir = Path(args.save_path) / "covariances" / args.cluster_type
    cov_dir.mkdir(parents=True, exist_ok=True)
    
    layers_to_compute = []
    for layer_idx in args.layers_to_compress:
        sigma_filename = f"{model_name}_SigmaMatrix_layer_{layer_idx}_{args.cluster_type}_wsamples_{args.whitening_nsamples}.pt"
        if (cov_dir / sigma_filename).exists():
            print(f"已找到层 {layer_idx} 的缓存, 跳过。")
        else:
            layers_to_compute.append(layer_idx)
    
    if layers_to_compute:
        print(f"\n{'='*20} Computing covariances for layers: {layers_to_compute} {'='*20}")
        
        newly_computed_covariances = profle_svdllm_low_resource(
            model=model, model_config=model_config, calib_loader=calib_data, dev='cuda',
            selected_layers=layers_to_compute, cluster_type=args.cluster_type,
            cluster_info=None, args=args
        )
        
        for layer_idx, cov_matrix in newly_computed_covariances.items():
            sigma_filename = f"{model_name}_SigmaMatrix_layer_{layer_idx}_{args.cluster_type}_wsamples_{args.whitening_nsamples}.pt"
            cache_path = cov_dir / sigma_filename
            torch.save(cov_matrix, cache_path)
            print(f"已保存协方差矩阵: {cache_path}")
    else:
        print("\n所有需要的协方差矩阵均已从缓存加载。")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
def run_whitening(args):

    # 步骤 2: 加载配置信息
    print("\n--- 启动张量分解 ---")
    model_config = get_model_config(args.model_path)
    temp_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_hidden_config = temp_config.to_dict()
    model_hidden_config['hidden_size'] = model_hidden_config.get('hidden_size') or model_hidden_config.get('d_model')
    model_hidden_config['intermediate_size'] = model_hidden_config.get('intermediate_size') or model_hidden_config.get('moe_intermediate_size')
    
    num_experts_from_config = temp_config.num_local_experts
    if num_experts_from_config is None:
        raise ValueError("无法从模型配置中确定 `num_local_experts`。")
    total_moe_layers = int(getattr(temp_config, "num_hidden_layers"))
    effective_local_ratio = _resolve_effective_local_ratio(args, total_moe_layers)
    args.effective_local_ratio = effective_local_ratio
    del temp_config

    print(
        f"[Ratio] scope={args.ratio_scope}, requested={args.ratio}, "
        f"effective_local={effective_local_ratio:.6f}, total_moe_layers={total_moe_layers}, "
        f"selected={len(args.layers_to_compress)}"
    )
    
    model_name_str = Path(args.model_path).name

    for layer_idx in args.layers_to_compress:
        current_layer_mode = args.cluster_type
        if current_layer_mode != 'global':
            raise NotImplementedError("此脚本目前仅为 'global' 模式配置。")

        # save_root_dir = Path(args.save_path) / "decomposition_results" / model_name_str / "adaptive"
        save_root_dir = Path(args.save_path) / "decomposition_results" / model_name_str / args.cluster_type / args.decomposition_method / args.whiten_type
        ratio_scope_dir = (
            "ratio_scope_local"
            if args.ratio_scope == "local"
            else f"ratio_scope_global_eff_{_ratio_to_str(effective_local_ratio)}"
        )
        save_root_dir = save_root_dir / ratio_scope_dir
        if args.rank_policy != "default":
            save_root_dir = save_root_dir / f"policy_{args.rank_policy}"
        save_root_dir.mkdir(parents=True, exist_ok=True)        
        print(f"\n分解结果将保存至: {save_root_dir}")
        
        loaded_stats = None
        if args.whiten_type != 'none':
            cov_dir = Path(args.save_path) / "covariances" / current_layer_mode        
            sigma_filename = f"{model_name_str}_SigmaMatrix_layer_{layer_idx}_{current_layer_mode}_wsamples_{args.whitening_nsamples}.pt"
            cache_path = cov_dir / sigma_filename
            if not cache_path.exists(): raise FileNotFoundError(f"协方差矩阵文件未找到: {cache_path}。")
            loaded_stats = torch.load(cache_path, map_location='cpu')
        
        print(f"\n-- 正在处理第 {layer_idx} 层 --")
        model, _ = load_cpu_model(args.model_path)
        moe_layer_name = model_config.moe_layer_pattern.format(layer_idx)
        
        num_experts = num_experts_from_config
        d_in_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_in_key']]
        d_out_up = model_hidden_config[FFN_ROLE_CONFIGS['up']['d_out_key']]
        
        try:
            if args.rank_policy == "equal":
                ranks_up = calculate_tucker_ranks_equal(num_experts, d_out_up, d_in_up, effective_local_ratio, fix_r0=True)
            else:
                ranks_up = calculate_tucker_ranks(num_experts, d_out_up, d_in_up, effective_local_ratio)
            ranks_down = [ranks_up[0], ranks_up[2], ranks_up[1]]
            print(f"  - 计算得到的Ranks: up层 {ranks_up}, down层 {ranks_down}")
        except KeyError:
            print(f"警告: 层 {layer_idx} 的 rank 计算失败，跳过此层。")
            continue
                                        
        for role, param_name in model_config.role_map.items():
            ratio_tag = _ratio_to_str(effective_local_ratio)
            if args.rank_policy == "default":
                save_filename = f"layer_{layer_idx}_{role}_ratio_{ratio_tag}_{args.whiten_type}.pt"
            else:
                save_filename = f"layer_{layer_idx}_{role}_ratio_{ratio_tag}_{args.whiten_type}_policy_{args.rank_policy}.pt"
            final_save_path = save_root_dir / save_filename
            if final_save_path.exists():
                print(f"已存在分解结果: {final_save_path}, 跳过。")
                continue
            
            print(f"\n--- 正在压缩: 第 {layer_idx} 层, 角色: '{role}' ---")
            print(f"压缩结果将存至：{final_save_path}")
            
            expert_weights = [dict(model.named_parameters())[f"{moe_layer_name}.experts.{i}.{param_name}.weight"].data.cpu() for i in range(num_experts)]
            W = torch.stack(expert_weights, dim=0)
            
            ranks = ranks_up if role in ['gate', 'up'] else ranks_down
            W_to_decompose = W
            Cholesky_matrices = {'S2': None, 'S3': None}

            if args.whiten_type != 'none':
                Sigma2, Sigma3 = get_covariances_for_role(role, args.whiten_type, loaded_stats)
                W_to_decompose, Cholesky_matrices = whiten_tensor(W, Sigma2, Sigma3, eps=1e-3, dev='cpu')
            
            # core, factors = hosvd_decomposition(W_to_decompose, ranks)
            core, factors = tucker_decomposition(W_to_decompose, ranks, method=args.decomposition_method)
            torch.save({'core': core, 'factors': factors, 'Cholesky': Cholesky_matrices}, final_save_path)
            print(f"已保存分解组件至: {final_save_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tucker-MoE Compression Framework")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_path", type=str, default="./output", help="输出路径")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="用于协方差计算的校准数据集")
    parser.add_argument("--whitening_nsamples", type=int, default=128, help="用于协方差计算的样本数")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="模型序列长度")
    parser.add_argument("--cluster_type", type=str, default="global", choices=["global"], help="目前只支持 'global' 模式")
    parser.add_argument("--whiten_type", type=str, required=True, choices=["input", "output", "both", "none"], help="白化类型")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--ratio_scope", type=str, default="local", choices=["local", "global"], help="`local`: ratio作用于被压缩层；`global`: ratio视为全部MoE层的全局目标")
    
    parser.add_argument("--ppl_datasets",type=str,nargs='+',default=["wikitext2", "ptb", "c4"], help="Datasets to use for PPL evaluation")   
    parser.add_argument("--eval_tasks", type=str,nargs='+', default=["openbookqa", "arc_easy", "winogrande", "arc_challenge", "piqa", "mathqa", "hellaswag"], help="Task names for lm-eval")      
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--lm_eval_batch_size", type=int, default=32, help="Batch size for lm-eval evaluations")
    parser.add_argument("--layers_to_compress", type=int, default=None, nargs='+')

    parser.add_argument("--global_layers", type=int, nargs='+', default=None, help="List of layer indices to be treated as 'global' in 'mixed' mode.")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters/groups for experts (used with cluster_type='group' or 'mixed')")
    parser.add_argument("--important_layers", type=int, nargs='+', default=None, help="手动指定的重要性层列表")
    parser.add_argument("--importance_factor", type=float, default=1.5, help="重要层相对于次要层的Ratio倍数")

    parser.add_argument("--run_eval", type=bool, default=False) 
    parser.add_argument("--decomposition_method", type=str, default="svd", choices=["svd", "qr", "rand"], help="用于Tucker分解的基底生成方法: svd, qr (QR分解), 或 rand (随机SVD)")
    parser.add_argument("--rank_policy", type=str, default="default", choices=["default", "equal"], help="Rank allocation policy")
    
    args = parser.parse_args()
    
    
    # args.ppl_datasets = [ "ptb","wikitext2"]
    # args.eval_tasks = ["winogrande", "openbookqa", "arc_easy",  "arc_challenge", "piqa"]
    
    
    print(f"{args.layers_to_compress=}")
    
    print(f"传入参数: {args}")
    
    
    # 计算协方差矩阵存在
    if args.whiten_type != 'none':
        run_covariances(args)    
    
    start_time = time.time()
    run_whitening(args)
    end_time = time.time()
    
    
    # print(f"白化总耗时: {end_time - start_time:.4f} 秒")
    
    if args.run_eval:
        run_evaluation.main(args)
