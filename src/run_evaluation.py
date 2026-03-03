# /home/Anonymous/TD-MoE/src/run_evaluation.py

import argparse
import torch
import json
from pathlib import Path
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils import load_cpu_model,load_fp16_model
from evaluator import run_lm_eval, ppl_eval_sharing
from config import get_model_config, set_layer_by_name, get_layer_by_name
from components.tucker_mixtral import MixtralTuckerDecomposedMoE
# from components.custom_mixtral import CustomMixtralDecoderLayer
from run_whitening import allocate_ratios_by_importance
    
def main(args):
    print(f"--- 启动评估 ---")
    print(f"Cluster Type: {args.cluster_type.upper()}, Whiten Type: {args.whiten_type.upper()}")
    
    print("\n--- 1. 加载基准模型 ---")
    model, tokenizer = load_fp16_model(args.model_path)
    model.eval()

    model_config = get_model_config(args.model_path)
    model_name = Path(args.model_path).name
    
    eval_save_dir = Path(args.save_path) / "evaluation_results" / model_name
    eval_save_dir.mkdir(parents=True, exist_ok=True)    

    cluster_data = None
    if args.cluster_type in ["group", "mixed"]:
        cluster_file_path = Path(args.save_path) / "cluster_results" / model_name / f"kmeans_groups_{args.num_clusters}.json"
        if not cluster_file_path.exists():
            raise FileNotFoundError(f"评估 '{args.cluster_type}' 模式需要聚类文件，但在路径 '{cluster_file_path}' 未找到。请先运行 run_clustering.py。")
        
        print(f"为 '{args.cluster_type}' 模式加载聚类数据: {cluster_file_path}")
        with open(cluster_file_path, 'r') as f:
            cluster_data = json.load(f)["cluster_groups"]
            
    adaptive_ratios = {}
    if args.important_layers:
        print("\n--- 启动手动自适应Ratio模式 ---")
        adaptive_ratios = allocate_ratios_by_importance(
            args.ratio, 
            args.layers_to_compress, 
            args.important_layers, 
            args.importance_factor
        )
        print("--- 自适应Ratio分配结果 (用于查找文件名) ---")            

    # --- 原始模型评估 ---
    experiment_name = "ppl_origin.txt"
    ppl_origin_path = eval_save_dir / experiment_name
    if not ppl_origin_path.exists():
        print(f"{ppl_origin_path} 不存在...")
        print(f"\n--- 原始模型跑ppl ---")
        origin_result = ppl_eval_sharing(
            model, tokenizer, "cuda", experiment_name=experiment_name,
            datasets=args.ppl_datasets, model_seq_len=args.model_seq_len,
            batch_size=args.eval_batch_size,
        )
        print("\n--- 原始模型评估结果 ---")
        print(origin_result)
        with open(ppl_origin_path, "w") as f:
            f.write(origin_result)
            f.write(f"\nArgs:\n{json.dumps(vars(args), indent=2)}")
        print(f"原始模型评估报告已保存至: {ppl_origin_path}")
    else:
        print(f"\n--- 已发现 {ppl_origin_path}，跳过原始模型评估 ---")
    
    task_names=["openbookqa", "arc_easy", "winogrande","arc_challenge", "piqa", "mathqa", "hellaswag"]
    acc_origin_path = eval_save_dir / f"acc_origin.csv"
    if not acc_origin_path.exists():
        print(f"\n--- 原始模型跑acc ---")
        results = run_lm_eval(model, tokenizer, batch_size=16,task_names=task_names, output_csv=acc_origin_path)
        print(f"{results}")
        results.to_csv(acc_origin_path)
    else:
        print(f"\n--- 已发现 {acc_origin_path}，跳过原始模型评估 ---")
    
    print(f"\n--- 2. 开始替换MoE模块，目标层: {args.layers_to_compress} ---")
    for layer_idx in tqdm(args.layers_to_compress, desc="替换模型层"):
        decoder_layer_name = f"model.layers.{layer_idx}"
        original_decoder_layer = get_layer_by_name(model, decoder_layer_name)
        print(f"正在处理第 {layer_idx} 层...")

        # 1. 确定当前层的处理模式
        current_layer_mode = args.cluster_type
        if args.cluster_type == 'mixed':
            if args.global_layers and layer_idx in args.global_layers:
                current_layer_mode = 'global'
                print(f"  - 模式: mixed -> 当前层 {layer_idx} 按 GLOBAL 模式处理")
            else:
                current_layer_mode = 'group'
                print(f"  - 模式: mixed -> 当前层 {layer_idx} 按 GROUP 模式处理")

        # 2. 根据当前模式设置正确的加载路径
        # 优先匹配 run_tucker.py 的目录结构: .../<cluster_type>/<decomposition_method>/<whiten_type>
        # 同时兼容旧目录结构: .../<cluster_type>/<whiten_type>
        if args.important_layers and current_layer_mode == 'global':
            base_results_path = Path(args.save_path) / "decomposition_results" / model_name / current_layer_mode / "adaptive"
        else:
            modern_base_path = (
                Path(args.save_path)
                / "decomposition_results"
                / model_name
                / current_layer_mode
                / args.decomposition_method
                / args.whiten_type
            )
            legacy_base_path = (
                Path(args.save_path)
                / "decomposition_results"
                / model_name
                / current_layer_mode
                / args.whiten_type
            )
            base_results_path = modern_base_path if modern_base_path.exists() else legacy_base_path
        
        # 3. 为当前层加载分解数据
        layer_decomposition_data = {}
        
        # <--- 修改/新增 START: 使用 current_layer_mode 判断 --->
        if current_layer_mode == "group":
            layer_key = f"layer_{layer_idx}"
            if not cluster_data or layer_key not in cluster_data:
                raise ValueError(f"警告: 在聚类文件中找不到第 {layer_idx} 层的信息。")
            
            for group_name in cluster_data[layer_key].keys():
                group_data = {}
                for role, role_param_name in model_config.role_map.items():
                    comp_filename = f"layer_{layer_idx}_{role}_{group_name}_ratio_{args.ratio}_{args.whiten_type}.pt"
                    comp_path = base_results_path / comp_filename
                    if not comp_path.exists():
                        raise FileNotFoundError(f"缺少分组分解文件: {comp_path}")
                    
                    print(f"  - 加载 {group_name} 的 '{role}' 组件...")
                    group_data[role_param_name] = torch.load(comp_path, map_location='cpu')
                layer_decomposition_data[group_name] = group_data
        
        elif current_layer_mode == "global":
            for role, role_param_name in model_config.role_map.items():                
                
                # 自适应模式的文件名
                if args.important_layers:
                    comp_filename = f"layer_{layer_idx}_{role}_ratio_adp_{args.ratio}_{args.whiten_type}.pt"
                else:
                    # 固定模式的文件名
                    comp_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}.pt"
                    
                 # comp_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}_balanced.pt"                                    
                comp_path = base_results_path / comp_filename
                
                if not comp_path.exists():
                    # comp_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}.pt"
                    # comp_path = base_results_path / comp_filename                    
                    raise FileNotFoundError(f"警告: 在层 {layer_idx} 缺少分解文件，路径: {comp_path}。")
                
                print(f"从 {comp_path} 加载分解组件...")
                layer_decomposition_data[role_param_name] = torch.load(comp_path, map_location='cpu')
                print(f"第 {layer_idx} 层分解组件")
                print(f"core.shape = {layer_decomposition_data[role_param_name]['core'].shape}")
                print(f"factor[0].shape = {layer_decomposition_data[role_param_name]['factors'][0].shape}")
                print(f"factor[1].shape = {layer_decomposition_data[role_param_name]['factors'][1].shape}")
                print(f"factor[2].shape = {layer_decomposition_data[role_param_name]['factors'][2].shape}")         
                

        # 4. 创建你的自定义 Tucker MoE 模块实例
        DecomposedMoEClass = model_config.decomposed_class
        try:
            compressed_moe_block = DecomposedMoEClass(
                config=model.config,
                original_moe=original_decoder_layer.block_sparse_moe,
                decompose_data=layer_decomposition_data,
                layer_idx=layer_idx,
                cluster_type=current_layer_mode,
                cluster_info=cluster_data.get(f"layer_{layer_idx}") if current_layer_mode == 'group' and cluster_data else None,
                model_dtype=model.dtype
            )

            original_decoder_layer.block_sparse_moe = compressed_moe_block
        except:
            compressed_moe_block = DecomposedMoEClass(
                config=model.config,
                original_moe=original_decoder_layer.mlp,
                decompose_data=layer_decomposition_data,
                layer_idx=layer_idx,
                cluster_type=current_layer_mode,
                cluster_info=cluster_data.get(f"layer_{layer_idx}") if current_layer_mode == 'group' and cluster_data else None,
                model_dtype=model.dtype
            )

            original_decoder_layer.mlp = compressed_moe_block            
            
            
        print(f"第 {layer_idx} 层已成功替换。")

    gc.collect()
    torch.cuda.empty_cache()
    print(model)

    layers_str = "_".join(map(str, args.layers_to_compress))
    experiment_name = f"{args.dataset}_ratio_{args.ratio}_layers_{layers_str}_{args.cluster_type}_{args.whiten_type}"
    
    if args.important_layers:
        experiment_name = f"{args.dataset}_adp_factor_{args.importance_factor}_ratio_{args.ratio}_layers_{layers_str}_{args.cluster_type}_{args.whiten_type}"
    
    ppl_save_path = eval_save_dir / f"ppl_{experiment_name}.txt"
    
    print(f"\n--- 开始困惑度评估: ppl_{experiment_name} ---")
    if not ppl_save_path.exists():        
        result_str = ppl_eval_sharing(
            model, tokenizer, "cuda", experiment_name=experiment_name,
            datasets=args.ppl_datasets, model_seq_len=args.model_seq_len,
            batch_size=args.eval_batch_size,
        )
        print("\n--- 评估结果 ---")
        print(result_str)
        with open(ppl_save_path, 'w') as f:
            f.write(result_str)
            f.write(f"\nArgs:\n{json.dumps(vars(args), indent=2)}")
    else:
        print(f"{ppl_save_path} 已存在，跳过...")
    
    print(f"\n--- 开始准确率评估: acc_{experiment_name} ---")
    output_csv = f"{eval_save_dir}/acc_{experiment_name}.csv"
    results = run_lm_eval(model, tokenizer, batch_size=args.lm_eval_batch_size,task_names=args.eval_tasks, output_csv=output_csv)
    print(f"\n评估报告已保存至: {eval_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tucker-Compressed MoE Model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--lm_eval_batch_size", type=int, default=192)
    
    parser.add_argument("--ppl_datasets",type=str,nargs='+',default=["wikitext2", "ptb", "c4"], help="Datasets to use for PPL evaluation")    
    parser.add_argument("--eval_tasks", type=str,nargs='+', default=["openbookqa", "arc_easy", "winogrande", "arc_challenge", "piqa", "mathqa", "hellaswag"], help="Task names for lm-eval")  
    
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--cluster_type", type=str, default="global", choices=["global", "group", "mixed"])
    parser.add_argument("--decomposition_method", type=str, default="svd", choices=["svd", "qr", "rand"])
    parser.add_argument("--whiten_type", type=str, required=True, choices=["input", "output", "both", "none"])
    parser.add_argument("--layers_to_compress", type=int, nargs='+', required=True)
    parser.add_argument("--global_layers", type=int, nargs='+', default=None, help="List of layer indices to be treated as 'global' in 'mixed' mode.")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters/groups for experts (used with cluster_type='group' or 'mixed')")
    parser.add_argument("--important_layers", type=int, nargs='+', default=None, help="手动指定的重要性层列表")
    parser.add_argument("--importance_factor", type=float, default=1.5, help="重要层相对于次要层的Ratio倍数")

    
    args = parser.parse_args()
    print(f"传入参数: {args}")
    main(args)
