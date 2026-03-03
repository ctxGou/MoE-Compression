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
from tucker_decomposition import calculate_tucker_ranks, whiten_tensor, hosvd_decomposition,calculate_tucker_ranks_balanced
from utils import load_cpu_model,load_fp16_model
from tqdm import tqdm
from config import get_model_config, set_layer_by_name, get_layer_by_name
from evaluator import run_lm_eval, ppl_eval_sharing


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
    
def run_whitening(args,rank_df):

    # 步骤 2: 加载配置信息
    print("\n--- 启动张量分解 ---")
    model_config = get_model_config(args.model_path)
    temp_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_hidden_config = temp_config.to_dict()
    model_hidden_config['hidden_size'] = model_hidden_config.get('hidden_size') or model_hidden_config.get('d_model')
    model_hidden_config['intermediate_size'] = model_hidden_config.get('intermediate_size') or model_hidden_config.get('moe_intermediate_size')
    
    try:
        num_experts_from_config = temp_config.num_local_experts
    except:
        num_experts_from_config = temp_config.n_routed_experts
    
    # if num_experts_from_config is None:
    #     raise ValueError("无法从模型配置中确定 `num_local_experts`。")
    # del temp_config
    

    # if num_experts_from_config is None:
    #     # 如果都找不到，抛出更详细的错误
    #     raise ValueError(f"无法从模型配置中确定专家数量。已尝试 {expert_keys_to_try}。")
    
    # print(f"成功从模型配置中通过 '{found_key}' 获取专家数量: {num_experts_from_config}")
    # del temp_config   
    
    model_name_str = Path(args.model_path).name

    for layer_idx in args.layers_to_compress:
        current_layer_mode = args.cluster_type
        if current_layer_mode != 'global':
            raise NotImplementedError("此脚本目前仅为 'global' 模式配置。")

        # save_root_dir = Path(args.save_path) / "decomposition_results" / model_name_str / args.cluster_type / args.whiten_type 
        save_root_dir = Path(args.save_path) / "decomposition_results" / model_name_str / "adaptive"
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
            rank_info = rank_df.loc[layer_idx]
            params_full = float(rank_info['params_full'])
            params_used = float(rank_info['params_used'])
            current_ratio = float(rank_info['compression_ratio'])
                                    
            print(f"  - 从CSV获取目标: params_used={params_used:.0f}, params_full={params_full:.0f} -> Target Ratio: {current_ratio:.4f}")
            
            ranks_up = calculate_tucker_ranks(num_experts, d_out_up, d_in_up, current_ratio)
            ranks_down = [ranks_up[0], ranks_up[2], ranks_up[1]]
            print(f"  - 计算得到的Ranks: up层 {ranks_up}, down层 {ranks_down}")
        except KeyError:
            print(f"警告: 在 {args.ratio_alloc_csv_path} 中找不到第 {layer_idx} 层的压缩率信息，跳过此层。")
            continue
                                        
        for role, param_name in model_config.role_map.items():
            save_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}.pt"
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
            
            core, factors = hosvd_decomposition(W_to_decompose, ranks)
            torch.save({'core': core, 'factors': factors, 'Cholesky': Cholesky_matrices}, final_save_path)
            print(f"已保存分解组件至: {final_save_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        
def run_evaluation(args):
    print(f"--- 启动评估 ---")
    print(f"Cluster Type: {args.cluster_type.upper()}, Whiten Type: {args.whiten_type.upper()}")
    
    print("\n--- 1. 加载基准模型 ---")
    model, tokenizer = load_fp16_model(args.model_path)
    model.eval()

    model_config = get_model_config(args.model_path)
    model_name = Path(args.model_path).name
    
    eval_save_dir = Path(args.save_path) / "evaluation_results" / model_name
    eval_save_dir.mkdir(parents=True, exist_ok=True)    
            
    print(f"\n--- 2. 开始替换MoE模块，目标层: {args.layers_to_compress} ---")
    for layer_idx in tqdm(args.layers_to_compress, desc="替换模型层"):
        # decoder_layer_name = f"model.layers.{layer_idx}"
        # original_decoder_layer = get_layer_by_name(model, decoder_layer_name)
        # print(f"正在处理第 {layer_idx} 层...")
        
        full_moe_pattern = model_config.moe_layer_pattern
        try:
            decoder_layer_pattern, moe_block_name = full_moe_pattern.rsplit('.', 1)
        except ValueError:
            raise ValueError(f"无法从 moe_layer_pattern '{full_moe_pattern}' 中解析出解码层和MoE模块名。")

        # 使用解析出的信息
        decoder_layer_name = decoder_layer_pattern.format(layer_idx)
        original_decoder_layer = get_layer_by_name(model, decoder_layer_name)
        
        print(f"正在处理第 {layer_idx} 层 (MoE模块: {moe_block_name})...")
        
        

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
        base_results_path = Path(args.save_path) / "decomposition_results" / Path(args.model_path).name / "adaptive"
        # base_results_path = Path(args.save_path) / "decomposition_results" / Path(args.model_path).name / current_layer_mode / args.whiten_type
        
        # 3. 为当前层加载分解数据
        layer_decomposition_data = {}
                
        if current_layer_mode == "global":
            for role, role_param_name in model_config.role_map.items():                
                
                # comp_filename = f"layer_{layer_idx}_{role}_{args.whiten_type}_ratio_{args.ratio}.pt"       
                comp_filename = f"layer_{layer_idx}_{role}_ratio_{args.ratio}_{args.whiten_type}.pt"                                                           
                comp_path = base_results_path / comp_filename
                
                if not comp_path.exists():              
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
        original_moe = getattr(original_decoder_layer, moe_block_name)
        
        compressed_moe_block = DecomposedMoEClass(
            config=model.config,
            # original_moe=original_decoder_layer.block_sparse_moe,
            original_moe=original_moe,
            decompose_data=layer_decomposition_data,
            layer_idx=layer_idx,
            cluster_type=current_layer_mode,
            cluster_info=None,
            model_dtype=model.dtype
        )

        setattr(original_decoder_layer, moe_block_name, compressed_moe_block)
        # original_decoder_layer.block_sparse_moe = compressed_moe_block
        print(f"第 {layer_idx} 层已成功替换。")

    gc.collect()
    torch.cuda.empty_cache()
    print(model)

    layers_str = "_".join(map(str, args.layers_to_compress))
    experiment_name = f"{args.dataset}_ratio_{args.ratio}_layers_{layers_str}_{args.cluster_type}_{args.whiten_type}"
    
    # if args.important_layers:
    #     experiment_name = f"{args.dataset}_adp_factor_{args.importance_factor}_ratio_{args.ratio}_layers_{layers_str}_{args.cluster_type}_{args.whiten_type}"
    
    ppl_save_path = eval_save_dir / f"ppl_adp_{experiment_name}.txt"
    
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
    output_csv = f"{eval_save_dir}/acc_adp_{experiment_name}.csv"
    results = run_lm_eval(model, tokenizer, batch_size=16,task_names=args.eval_tasks, output_csv=output_csv)
    print(f"\n评估报告已保存至: {eval_save_dir}")        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tucker-MoE Compression Framework")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--save_path", type=str, default="./output", help="输出路径")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="用于协方差计算的校准数据集")
    parser.add_argument("--whitening_nsamples", type=int, default=128, help="用于协方差计算的样本数")
    parser.add_argument("--model_seq_len", type=int, default=2048, help="模型序列长度")
    parser.add_argument("--cluster_type", type=str, default="global", choices=["global"], help="目前只支持 'global' 模式")
    parser.add_argument("--whiten_type", type=str, required=True, choices=["input", "output", "both", "none"], help="白化类型")
    parser.add_argument("--ratio_alloc_csv_path", type=str,  help="ratio 分配的 CSV 文件路径")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--ppl_datasets",type=str,nargs='+',default=["wikitext2", "ptb", "c4"], help="Datasets to use for PPL evaluation")   
    parser.add_argument("--eval_tasks", type=str,nargs='+', default=["openbookqa", "arc_easy", "winogrande", "arc_challenge", "piqa", "mathqa", "hellaswag"], help="Task names for lm-eval")      
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--layers_to_compress", type=int, default=None, nargs='+')
    # parser.add_argument("--run_whitening", type=bool, default=True) 
    # parser.add_argument("--run_eval", type=bool, default=False) 
    
        
    args = parser.parse_args()
    
    # args.ratio_alloc_csv_path = f"./rank_allocation/outputs_{args.ratio}.csv"
    # args.ratio_alloc_csv_path = f"./rank_allocation/outputs_0.2_ori.csv"
    
    args.ppl_datasets = [ "ptb","wikitext2"]
    args.eval_tasks = ["winogrande", "openbookqa", "arc_easy",  "arc_challenge", "piqa"]
    
    print(f"\n--- 从CSV文件加载每层的压缩率目标: {args.ratio_alloc_csv_path} ---")
    if not os.path.exists(args.ratio_alloc_csv_path):
        raise FileNotFoundError(f"Rank 分配文件未找到: {args.ratio_alloc_csv_path}")
    rank_df = pd.read_csv(args.ratio_alloc_csv_path)
    rank_df.set_index('layer', inplace=True)  
    
    print(f"{rank_df=}")    
    
    if not args.layers_to_compress:
        args.layers_to_compress = list(rank_df[rank_df['compression_ratio']>0].index)
    
    # layers_to_compress=[18, 20, 21, 22, 23, 24, 25, 26, 27]
    # args.layers_to_compress = [18, 20, 21]
    # args.layers_to_compress = [22, 23, 24]
    # args.layers_to_compress = [25, 26, 27]
    
    print(f"{args.layers_to_compress=}")
    
    print(f"传入参数: {args}")
    
    # # 计算协方差矩阵
    # if args.whiten_type != 'none':
    #     run_covariances(args)    
    
    # start_time = time.time()
    # run_whitening(args,rank_df)
    # end_time = time.time()
    
    # print(f"白化总耗时: {end_time - start_time:.4f} 秒")
    

    run_evaluation(args)