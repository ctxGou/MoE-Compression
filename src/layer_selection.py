import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

import torch
import torch.nn as nn
import gc
import json
from pathlib import Path
from tqdm import tqdm
import copy
import time
import math
import subprocess  # 导入subprocess模块

# 确保从您的项目中正确导入所需模块
from utils import load_fp16_model, get_test_data 
from config import get_model_config, get_layer_by_name
from components.tucker_mixtral import MixtralTuckerDecomposedMoE 

# ==============================================================================
# 本地PPL评估函数 (此部分不变)
# ==============================================================================
@torch.no_grad()
def _evaluate_ppl_on_subset(model, data_subset):
    """在此脚本内部计算给定数据子集的PPL。"""
    model.eval()
    main_device = next(model.parameters()).device
    loss_fct = nn.CrossEntropyLoss()
    
    nlls = []
    total_tokens = 0

    for batch in data_subset:
        batch = batch.to(main_device)
        shift_labels = batch[:, 1:].contiguous()
        tokens_in_batch = shift_labels.numel()
        total_tokens += tokens_in_batch

        outputs = model(batch)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        neg_log_likelihood = loss.float() * tokens_in_batch
        nlls.append(neg_log_likelihood)

    if total_tokens == 0:
        return float('inf')
        
    final_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return final_ppl.item()

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【新增函数】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
def run_decomposition_if_needed(
    layers_to_evaluate: list, model_path: str, model_name: str, save_path: str,
    ratio: float, whiten_type: str, model_config
):
    """
    预检查所有需要的分解文件。如果任何文件缺失，
    则调用 run_whitening.py 脚本来生成它们。
    """
    print("\n--- 预检查分解文件是否存在... ---")
    base_results_path = Path(save_path) / "decomposition_results" / model_name / "global"
    layers_to_decompose = []

    for layer_idx in layers_to_evaluate:
        is_missing = False
        for role in model_config.role_map.keys():
            # 【修复】移除 ratio 的 .2f 格式化
            filename_reg = f"layer_{layer_idx}_{role}_ratio_{ratio}_{whiten_type}.pt"
            path_reg = base_results_path / filename_reg
            
            adaptive_filename = f"layer_{layer_idx}_{role}_ratio_adp_{ratio}_{whiten_type}.pt"
            path_adp = base_results_path / "adaptive" / adaptive_filename

            if not path_reg.exists() and not path_adp.exists():
                print(f"  -> 缺失文件 for layer {layer_idx}, role {role}")
                is_missing = True
                break
        
        if is_missing:
            layers_to_decompose.append(str(layer_idx))

    if layers_to_decompose:
        print(f"\n--- 检测到 {len(layers_to_decompose)} 个层缺少分解文件。将自动调用 run_whitening.py ---")
        print(f"待处理的层: {', '.join(layers_to_decompose)}")

        # 【修复】使用脚本的绝对路径来避免路径问题
        src_dir = Path(__file__).parent.resolve()
        script_to_run = src_dir / "run_whitening.py"
        
        command = [
            "python", str(script_to_run),
            "--model_path", model_path,
            "--save_path", save_path,
            "--cluster_type", "global",
            "--ratio", str(ratio),
            "--whiten_type", whiten_type,
            "--layers_to_compress"
        ]
        command.extend(layers_to_decompose)
        
        command_str = " ".join(command)
        print(f"执行命令: {command_str}")
        
        try:
            subprocess.run(command_str, shell=True, check=True)
            print("--- run_whitening.py 执行完毕。---")
        except subprocess.CalledProcessError as e:
            print(f"\n--- ERROR: run_whitening.py 执行失败 ---")
            print(e)
            raise RuntimeError("自动分解步骤失败，请检查错误信息。")
    else:
        print("--- 所有必需的分解文件均已存在。---")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【新增函数】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def temp_compress_and_replace_layer(
    model, model_config, layer_idx, model_dtype,
    save_path: Path, model_name: str,
    load_decomposed_ratio: float, whiten_type_for_loading: str
):
    """通过加载预分解的组件来临时替换单个MoE层，并返回原始模块以便恢复。"""
    decoder_layer_name = f"model.layers.{layer_idx}"
    original_decoder_layer = get_layer_by_name(model, decoder_layer_name)
    original_moe_block = original_decoder_layer.block_sparse_moe

    base_results_path = Path(save_path) / "decomposition_results" / model_name / "global"
    
    layer_decomposition_data = {}
    print(f"  (层 {layer_idx}) 正在从磁盘加载 ratio={load_decomposed_ratio} 的分解结果...")

    for role, role_param_name in model_config.role_map.items():
        # 【修复】确保ratio格式化为两位小数，与run_whitening.py的保存逻辑一致
        filename = f"layer_{layer_idx}_{role}_ratio_{load_decomposed_ratio}_{whiten_type_for_loading}.pt"
        comp_path = base_results_path / filename

        if not comp_path.exists():
            adaptive_filename = f"layer_{layer_idx}_{role}_ratio_adp_{load_decomposed_ratio}_{whiten_type_for_loading}.pt"
            adaptive_path = base_results_path / "adaptive" / adaptive_filename
            if adaptive_path.exists():
                comp_path = adaptive_path
            else:
                 raise FileNotFoundError(
                    f"缺少预分解文件 for layer {layer_idx}, role '{role}'.\n"
                    f"尝试查找路径: {comp_path} 和 {adaptive_path}"
                )

        layer_decomposition_data[role_param_name] = torch.load(comp_path, map_location='cpu')
    
    DecomposedMoEClass = model_config.decomposed_class
    if not DecomposedMoEClass:
        raise ValueError(f"模型 '{model_config.model_name_pattern}' 在 config.py 中缺少 'decomposed_class' 定义。")

    print(f"  (层 {layer_idx}) 正在基于加载的数据构建 Tucker experts...")
    compressed_moe_block = DecomposedMoEClass(
        config=model.config, original_moe=original_moe_block,
        decompose_data=layer_decomposition_data, layer_idx=layer_idx,
        cluster_type='global', cluster_info=None, model_dtype=model_dtype
    )
    original_decoder_layer.block_sparse_moe = compressed_moe_block
    
    return original_moe_block

def select_important_layers_by_ppl(
    model_path: str, layers_to_evaluate: list, num_important_layers: int,
    save_path: str, proxy_dataset: str = "wikitext2",
    load_decomposed_ratio: float = 0.6, 
    whiten_type_for_loading: str = "output",
    model_seq_len: int = 2048, eval_batch_size: int = 1
):
    """通过加载预分解组件并测量PPL增量来选择重要层。"""
    model_name = Path(model_path).name
    selection_dir = Path(save_path) / "layer_selection" / model_name
    selection_dir.mkdir(parents=True, exist_ok=True)
    
    layers_str = "_".join(map(str, sorted(layers_to_evaluate)))
    selection_filename = f"selection_top{num_important_layers}_from_{len(layers_to_evaluate)}layers_loaded_ratio{load_decomposed_ratio}.json"
    selection_cache_path = selection_dir / selection_filename

    if selection_cache_path.exists():
        print(f"--- 发现已缓存的层选择文件: {selection_cache_path} ---")
        with open(selection_cache_path, 'r') as f:
            result = json.load(f)
        print("加载重要层列表:", result["important_layers"])
        return result["important_layers"]

    print("\n--- 开始自动化层重要性选择 (快速加载+内存优化模式) ---")
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼【核心修改：增加预检查和自动分解】▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 在加载模型前，先检查并生成所需文件
    model_config_temp = get_model_config(model_path)
    run_decomposition_if_needed(
        layers_to_evaluate=layers_to_evaluate,
        model_path=model_path,
        model_name=model_name,
        save_path=save_path,
        ratio=load_decomposed_ratio,
        whiten_type=whiten_type_for_loading,
        model_config=model_config_temp
    )
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲【核心修改：增加预检查和自动分解】▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 一次性加载模型和数据
    print(f"模型: {model_name}\n待评估层: {layers_to_evaluate}")
    print(f"加载的分解率: {load_decomposed_ratio}, 白化类型: {whiten_type_for_loading}")
    
    model, tokenizer = load_fp16_model(model_path)
    model.eval()
    model_config = get_model_config(model_path)
    
    print("\n--- 正在准备代理评估数据集... ---")
    data_loader = get_test_data(proxy_dataset, tokenizer, seq_len=model_seq_len, batch_size=eval_batch_size)
    full_test_data_list = list(data_loader)
    num_batches_to_use = math.ceil(len(full_test_data_list) / 3)
    proxy_eval_data = full_test_data_list[:num_batches_to_use]
    print(f"已加载完整测试集 ({len(full_test_data_list)} batches)，将使用前 {len(proxy_eval_data)} batches (约1/3) 进行评估。")

    print("\n--- 步骤 1: 计算原始模型的基线PPL ---")
    baseline_ppl = _evaluate_ppl_on_subset(model, proxy_eval_data)
    print(f"'{proxy_dataset}' 1/3子集上的基线PPL: {baseline_ppl:.4f}")
    
    layer_sensitivities = {}

    for layer_idx in tqdm(layers_to_evaluate, desc="评估层敏感度"):
        print(f"\n--- 步骤 2: 评估层 {layer_idx} ---")
        decoder_layer = get_layer_by_name(model, f"model.layers.{layer_idx}")
        original_moe_block = None
        
        try:
            original_moe_block = temp_compress_and_replace_layer(
                model, model_config, layer_idx, model.dtype,
                save_path=Path(save_path),
                model_name=model_name,
                load_decomposed_ratio=load_decomposed_ratio,
                whiten_type_for_loading=whiten_type_for_loading
            )
            
            compressed_ppl = _evaluate_ppl_on_subset(model, proxy_eval_data)
            
            ppl_increase = compressed_ppl - baseline_ppl
            layer_sensitivities[layer_idx] = ppl_increase
            print(f"  - 加载压缩层 {layer_idx} 后的PPL: {compressed_ppl:.4f} (增量: {ppl_increase:.4f})")
        finally:
            if original_moe_block is not None:
                print(f"  - 正在恢复层 {layer_idx} 的原始模块...")
                decoder_layer.block_sparse_moe = original_moe_block
            gc.collect()
            torch.cuda.empty_cache()

    # ... (后续的排序、保存结果逻辑不变) ...
    sorted_layers = sorted(layer_sensitivities.items(), key=lambda item: item[1], reverse=True)
    important_layers = [layer[0] for layer in sorted_layers[:num_important_layers]]
    
    print("\n--- 层重要性排序 (基于PPL增量) ---")
    for idx, (layer, score) in enumerate(sorted_layers):
        label = "-> 重要" if idx < num_important_layers else "-> 次要"
        print(f"  - Layer {layer}: PPL Increase = {score:.4f} {label}")
        
    result_to_save = {
        "metadata": {
            "model_name": model_name, "layers_evaluated": sorted(layers_to_evaluate),
            "num_important_layers_selected": num_important_layers,
            "loaded_compression_ratio": load_decomposed_ratio,
            "proxy_dataset": proxy_dataset, "proxy_data_fraction": "1/3 of test set",
        },
        "important_layers": sorted(important_layers),
        "sensitivity_scores": {str(k): v for k, v in sorted(layer_sensitivities.items())}
    }
    with open(selection_cache_path, 'w') as f:
        json.dump(result_to_save, f, indent=4)
    print(f"\n--- 层选择结果已保存至: {selection_cache_path} ---")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return important_layers

if __name__ == "__main__":
    config = {
        "model_path": "./models/Mixtral-8x7B-v0.1",
        "save_path": "./output",
        # "layers_to_evaluate": [3, 5, 6, 7, 9, 12, 23, 24, 25],
        "layers_to_evaluate": [3, 5, 6, 7, 9, 10, 12, 13, 21, 22, 23, 24, 25, 26],
        "num_important_layers": 7,
        "load_decomposed_ratio": 0.2,
        "whiten_type_for_loading": "output",
        "proxy_dataset": "wikitext2",
        "model_seq_len": 2048,
        "eval_batch_size": 1
    }
    print("--- 使用硬编码的配置参数 (快速加载+内存优化模式) ---")
    print(json.dumps(config, indent=2))
    
    start_time = time.time()
    selected_layers = select_important_layers_by_ppl(**config)
    end_time = time.time()

    print(f"\n--- 自动化层选择完成 ---")
    print(f"选出的 {len(selected_layers)} 个重要层是: {sorted(selected_layers)}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")