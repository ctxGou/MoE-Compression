# /home/Anonymous/TD-MoE/src/run_covariances.py

import argparse
import os
import torch
import json
from pathlib import Path
import gc
import time
from utils import get_calib_train_data, load_cpu_model
from config import get_model_config
from data_collection import profle_svdllm_low_resource

def main(args):
    print("--- Step 1: Setting up and loading model ---")
    model, tokenizer = load_cpu_model(args.model_path)
    
    model_config = get_model_config(args.model_path)
    model_name = Path(args.model_path).name

    cluster_data = None
    if args.cluster_type in ["group", "mixed"]:
        cluster_file_path = Path(args.save_path) / "cluster_results" / model_name / f"kmeans_groups_{args.num_clusters}.json"
        if not cluster_file_path.exists():
            raise FileNotFoundError(f"评估 '{args.cluster_type}' 模式需要聚类文件，但在路径 '{cluster_file_path}' 未找到。请先运行 run_clustering.py。")
        
        print(f"为 '{args.cluster_type}' 模式加载聚类数据: {cluster_file_path}")
        with open(cluster_file_path, 'r') as f:
            cluster_data = json.load(f).get("cluster_groups")
        if not cluster_data:
            raise ValueError(f"聚类文件 '{cluster_file_path}' 中缺少或空的 'cluster_groups' 键。")

    print("--- Step 2: Preparing calibration data ---")
    calib_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
    
    # <--- 修改/新增 START: 混合模式的核心逻辑 --->
    if args.cluster_type == 'mixed':
        print("\n--- Mixed Mode Detected: Processing layers in two stages ---")
        if not args.global_layers:
            raise ValueError("'mixed' mode requires '--global_layers' to be specified.")
        
        # 1. 拆分任务列表
        all_layers = set(args.layers_to_compress)
        global_layers = set(args.global_layers)
        group_layers = list(all_layers - global_layers)
        global_layers = list(global_layers)
        
        # 2. 处理 Global 任务
        if global_layers:
            print(f"\n--- Stage 1: Processing GLOBAL layers: {global_layers} ---")
            cov_dir_global = Path(args.save_path) / "covariances" / "global"
            cov_dir_global.mkdir(parents=True, exist_ok=True)
            
            layers_to_compute_global = [
                idx for idx in global_layers 
                if not (cov_dir_global / f"{model_name}_SigmaMatrix_layer_{idx}_global_wsamples_{args.whitening_nsamples}.pt").exists()
            ]

            if layers_to_compute_global:
                computed_covs = profle_svdllm_low_resource(
                    model, model_config, calib_data, 'cuda', layers_to_compute_global, 'global', None, args
                )
                for layer_idx, cov_matrix in computed_covs.items():
                    save_path = cov_dir_global / f"{model_name}_SigmaMatrix_layer_{layer_idx}_global_wsamples_{args.whitening_nsamples}.pt"
                    torch.save(cov_matrix, save_path)
                    print(f"Saved GLOBAL covariance for layer {layer_idx} to {save_path}")
            else:
                print("All required GLOBAL covariances are already cached.")

        # 3. 处理 Group 任务
        if group_layers:
            print(f"\n--- Stage 2: Processing GROUP layers: {group_layers} ---")
            cov_dir_group = Path(args.save_path) / "covariances" / "group"
            cov_dir_group.mkdir(parents=True, exist_ok=True)
            
            layers_to_compute_group = [
                idx for idx in group_layers
                if not (cov_dir_group / f"{model_name}_SigmaMatrix_layer_{idx}_group_wsamples_{args.whitening_nsamples}.pt").exists()
            ]

            if layers_to_compute_group:
                cluster_info_for_comp = {f"layer_{idx}": cluster_data.get(f"layer_{idx}") for idx in layers_to_compute_group}
                computed_covs = profle_svdllm_low_resource(
                    model, model_config, calib_data, 'cuda', layers_to_compute_group, 'group', cluster_info_for_comp, args
                )
                for layer_idx, cov_matrix in computed_covs.items():
                    save_path = cov_dir_group / f"{model_name}_SigmaMatrix_layer_{layer_idx}_group_wsamples_{args.whitening_nsamples}.pt"
                    torch.save(cov_matrix, save_path)
                    print(f"Saved GROUP covariance for layer {layer_idx} to {save_path}")
            else:
                print("All required GROUP covariances are already cached.")

    else: # 单一模式 (Global 或 Group)
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
            
            cluster_info_for_computation = None
            if args.cluster_type == "group" and cluster_data:
                cluster_info_for_computation = {f"layer_{idx}": cluster_data.get(f"layer_{idx}") for idx in layers_to_compute}
            
            newly_computed_covariances = profle_svdllm_low_resource(
                model=model, model_config=model_config, calib_loader=calib_data, dev='cuda',
                selected_layers=layers_to_compute, cluster_type=args.cluster_type,
                cluster_info=cluster_info_for_computation, args=args
            )
            
            for layer_idx, cov_matrix in newly_computed_covariances.items():
                sigma_filename = f"{model_name}_SigmaMatrix_layer_{layer_idx}_{args.cluster_type}_wsamples_{args.whitening_nsamples}.pt"
                cache_path = cov_dir / sigma_filename
                torch.save(cov_matrix, cache_path)
                print(f"Saved covariance for layer {layer_idx} to {cache_path}")
        else:
            print("\nAll layer covariances were loaded from cache. No computation needed.")
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covariance Matrix Calculation for MoE Models")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--whitening_nsamples", type=int, default=128)
    parser.add_argument("--model_seq_len", type=int, default=2048)
    # <--- 修改/新增 START: 增加 mixed 模式相关参数 --->
    parser.add_argument("--cluster_type", type=str, default="global", choices=["global", "group", "mixed"])
    parser.add_argument("--layers_to_compress", type=int, nargs='+', required=True)
    parser.add_argument("--global_layers", type=int, nargs='+', default=None, help="List of layer indices to be treated as 'global' in 'mixed' mode.")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters/groups for experts (used with cluster_type='group' or 'mixed')")
    # <--- 修改/新增 END --->
    
    args = parser.parse_args()
    print(f"传入参数: {args}")
    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.4f} 秒")
