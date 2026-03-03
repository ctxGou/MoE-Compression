# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from tqdm import tqdm
from pathlib import Path
import math
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ==============================================================================
#  ↓↓↓ 这是您提供的 Tucker 分解参数和秩计算函数，保持不变 ↓↓↓
# ==============================================================================

def tucker_params(E: int, d_out: int, d_in: int, rE: int, rO: int, rI: int) -> int:
    """计算 Tucker 分解张量的参数量。"""
    core = rE * rO * rI
    factor = E * rE + d_out * rO + d_in * rI
    return core + factor

def calculate_tucker_ranks(n0, n1, n2, ratio, fix_r0=True) -> List[int]:
    """根据目标压缩比例 ratio 估算 r1 和 r2。"""
    if not (0 < 1 - ratio < 1):
        return [n0, n1, n2]
    r0 = n0 if fix_r0 else None
    original_params = n0 * n1 * n2
    target_params = int(original_params * (1 - ratio))
    best_r1, best_r2 = n1, n2
    min_error = float('inf')
    a = r0 * n1 * n2
    b = n1**2 + n2**2
    c = n0 * r0 - target_params
    if a == 0:
        scale = -c / b if b != 0 else 0
    else:
        delta = b**2 - 4 * a * c
        scale = (-b + math.sqrt(max(0, delta))) / (2 * a)
    est_r1 = max(1, int(n1 * scale))
    est_r2 = max(1, int(n2 * scale))
    for r1 in range(max(1, est_r1 - 5), min(n1 + 1, est_r1 + 5)):
        numerator = target_params - (n0 * r0 + n1 * r1)
        denominator = r0 * r1 + n2
        if denominator <= 0:
            continue
        r2 = numerator // denominator
        if 1 <= r2 <= n2:
            compressed = tucker_params(n0, n1, n2, r0, r1, r2)
            error = abs(compressed - target_params)
            if error < min_error:
                min_error = error
                best_r1, best_r2 = r1, r2
    if best_r1 is None or best_r2 is None:
        return [n0, n1, n2]
    return [r0, best_r1, best_r2]


# ==============================================================================
#  ↓↓↓ 这是整合了您新逻辑的、逐层计算 Fisher 的函数 ↓↓↓
# ==============================================================================

def calculate_fisher_layerwise(
    model_path: str,
    model_type: str, # 新增参数，用于区分模型结构, e.g., 'mixtral' or 'deepseek'
    calib_dataset: str,
    calib_split: str,
    num_samples: int,
    batch_size: int,
    max_length: int
) -> Dict[int, float]:
    """
    Memory-efficient Fisher Information calculation (layer-wise, with gradient accumulation)
    """
    print("\n" + "="*50)
    print(" 开始逐层计算 Fisher Information (内存优化版) ".center(50, "="))
    
    device = "cuda:0" # 默认使用第一张可见卡来处理数据

    # 1. 加载模型和分词器
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.train() # 设置为 train 模式以确保梯度计算

    # 2. 加载和预处理校准数据
    print(f"Loading calibration dataset: {calib_dataset} ({calib_split})...")
    dataset = load_dataset(calib_dataset, calib_split, split="train")
    texts = dataset.select(range(num_samples))["text"]
    
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length
    )
    
    # 切分成多个小 batch
    input_batches = []
    for i in range(0, len(texts), batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in encodings.items()}
        input_batches.append(batch)
    print(f"Total {len(texts)} samples, split into {len(input_batches)} batches.")

    # 3. 逐层计算 Fisher 信息
    layer_importance = {}
    
    # --- 根据模型类型选择正确的层访问路径 ---
    try:
        transformer_layers = model.model.layers
        print(f"成功访问模型层列表，总层数: {len(transformer_layers)}")
    except AttributeError:
        print(f"错误：无法在 'model.model.layers' 找到模型层。请检查 '{model_path}' 的模型结构。")
        raise
        
    print("Computing Fisher information layer by layer (with accumulation)...")
    for i, layer in enumerate(tqdm(transformer_layers, desc="Processing Layers")):
        # 冻结所有层，只解冻当前层
        for param in model.parameters():
            param.requires_grad_(False)
        for param in layer.parameters():
            param.requires_grad_(True)

        score_total = 0.0
        
        # 在所有 batch 上累积梯度平方
        for batch in input_batches:
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            
            model.zero_grad()
            loss.backward()

            # 累加当前 batch 的梯度平方
            score = 0.0
            count = 0
            for param in layer.parameters():
                if param.grad is not None:
                    # 使用 .sum() 更符合 Fisher 的定义，但 .mean() 也可以衡量相对重要性
                    # 这里保持和您 DeepSeek 脚本一致的 .sum() 逻辑
                    score += (param.grad.detach().float() ** 2).sum().item()
            score_total += score

        layer_importance[i] = score_total / len(texts) # 按样本数进行平均
        print(f"[Layer {i}] Fisher={layer_importance[i]:.6e}")
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(" Fisher Information 计算完成 ".center(50, "="))
    print("="*50 + "\n")
    return layer_importance


# ==============================================================================
#  ↓↓↓ 这是自适应分配压缩率的核心函数，保持不变 ↓↓↓
# ==============================================================================
def allocate_tucker_ranks_adaptive(
    model_name: str,
    fisher_dict: Dict[int, float],
    layer_shapes: Dict[int, Tuple[int, int, int]],
    target_compression: float,
    save_dir: str,
    min_frac: float = 0.05,
    max_frac: float = 1.0,
    tol: float = 0.005,
    max_iter: int = 30,
    verbose: bool = False
) -> pd.DataFrame:
    # ... 此函数内容与之前完全相同，此处省略以保持简洁 ...
    layers = sorted(list(fisher_dict.keys()))
    n_layers = len(layers)
    full_params_total = sum(E * d_out * d_in for (E, d_out, d_in) in layer_shapes.values())
    target_params_total = full_params_total * (1 - target_compression)
    fishers = np.array([fisher_dict[l] for l in layers])
    
    fishers = fishers ** 0.5
    
    total_fisher = fishers.sum()
    fisher_weights = fishers / total_fisher if total_fisher > 0 else np.ones_like(fishers) / n_layers
    def one_pass(scale_global):
        results = []
        used_params_total = 0
        for i, l in enumerate(layers):
            E, d_out, d_in = layer_shapes[l]
            full_layer_params = E * d_out * d_in
            layer_target_params = (fisher_weights[i] * full_params_total) * scale_global
            layer_compression_ratio = np.clip(1 - (layer_target_params / full_layer_params), 1-max_frac, 1-min_frac)
            ranks = calculate_tucker_ranks(E, d_out, d_in, ratio=layer_compression_ratio)
            used = tucker_params(E, d_out, d_in, ranks[0], ranks[1], ranks[2])
            if used > full_layer_params:
                used = full_layer_params
            used_params_total += used
            results.append({
                "layer": l, "fisher": fisher_dict[l], "shape": (E, d_out, d_in),
                "params_full": full_layer_params, "params_used": used,
                "compression_ratio": 1 - used / full_layer_params if full_layer_params > 0 else 0,
            })
        realized_compression = 1 - used_params_total / full_params_total
        return pd.DataFrame(results), realized_compression
    low, high = 0.1, 5.0
    best_df, best_error = None, float("inf")
    for it in range(max_iter):
        scale_global = (low + high) / 2
        df_try, realized = one_pass(scale_global)
        err = target_compression - realized
        if abs(err) < abs(best_error):
            best_error, best_df = err, df_try
        if abs(err) <= tol:
            if verbose: print(f"达到容忍度, 提前退出。")
            break
        if realized < target_compression:
            high = scale_global
        else:
            low = scale_global
        if verbose:
            print(f"[Iter {it+1:02d}] scale_global={scale_global:.4f}, realized={realized:.4f}, target={target_compression:.4f}, err={err:+.4f}")
    df = best_df
    realized_compression = 1 - df["params_used"].sum() / df["params_full"].sum()
    print("\n" + "="*40)
    print(" 自适应Rank分配结果 ".center(40, "="))
    print(f"目标总压缩率: {target_compression:.2%}")
    print(f"实现总压缩率: {realized_compression:.2%} (误差 {realized_compression - target_compression:+.2%})")
    print("="*40 + "\n")
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"outputs_{model_name}_{target_compression}"
    csv_path = os.path.join(save_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"结果已保存至: {csv_path}")
    return df


# ==============================================================================
#  ↓↓↓ 脚本执行入口：所有配置请在这里修改 ↓↓↓
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 基础配置 ---
    MODEL_PATH = ""
    # 【重要】请根据你的模型选择 'deepseek' 或 'mixtral'
    MODEL_TYPE = 'mixtral'
    SAVE_DIR = "./rank_allocation/outputs"
    TARGET_COMPRESSION = 0.2

    # --- 2. Fisher 信息计算配置 ---
    FISHER_CONFIG = {
        "calib_dataset": "wikitext",
        "calib_split": "wikitext-2-raw-v1",
        "num_samples": 128,  # 总样本数
        "batch_size": 16,   # 每个小 batch 样本数，根据显存调整
        "max_length": 128,  # 序列长度
    }

    # --- 3. 模型 Shape 配置 ---
    # 请根据你的模型确认MoE层的 shape (experts, intermediate_size, hidden_size)
    if MODEL_TYPE == 'deepseek':
        # DeepSeek-MoE-16B 有28个MoE层 (0-27)
        layer_shapes = {i: (64, 14336, 4096) for i in range(28)}
    elif MODEL_TYPE == 'mixtral':
        # Mixtral-8x7B 有32个MoE层 (0-31)
        layer_shapes = {i: (8, 14336, 4096) for i in range(32)}
    else:
        raise ValueError(f"不支持的模型类型: {MODEL_TYPE}. 请在 'deepseek' 或 'mixtral' 中选择。")

    
    MODEL_NAME = Path(MODEL_PATH).name
    print(f"模型: {MODEL_NAME}, 类型: {MODEL_TYPE}, 目标压缩率: {TARGET_COMPRESSION:.2%}")

    # --- 4. 执行流程 ---
    # 检查是否存在缓存的 Fisher 信息文件
    fisher_cache_path = os.path.join(SAVE_DIR, f"fisher_info_{MODEL_NAME}.json")
    
    if os.path.exists(fisher_cache_path):
        print(f"从缓存加载 Fisher Information: {fisher_cache_path}")
        with open(fisher_cache_path, 'r') as f:
            # json的key是字符串，需要转回整型
            fisher_dict_str_keys = json.load(f)
            fisher_dict = {int(k): v for k, v in fisher_dict_str_keys.items()}
    else:
        # 如果没有缓存，则进行计算
        fisher_dict = calculate_fisher_layerwise(
            model_path=MODEL_PATH,
            model_type=MODEL_TYPE,
            **FISHER_CONFIG
        )
        # 保存为 JSON 文件，方便人类阅读和后续加载
        with open(fisher_cache_path, "w") as f:
            json.dump({str(k): float(v) for k, v in fisher_dict.items()}, f, indent=2)
        print(f"Fisher Information 已计算并保存至: {fisher_cache_path}")

    # 运行自适应分配算法
    df = allocate_tucker_ranks_adaptive(
        model_name=MODEL_NAME,
        fisher_dict=fisher_dict,
        layer_shapes=layer_shapes,
        target_compression=TARGET_COMPRESSION,
        save_dir=SAVE_DIR,
        verbose=True
    )

    print("\n--- 生成的分配方案 (前5行) ---")
    print(df.head())

    # --- 5. (可选) 可视化 Fisher 结果 ---
    plt.figure(figsize=(12, 6))
    layers = sorted(fisher_dict.keys())
    values = [fisher_dict[l] for l in layers]
    plt.bar(layers, values, color="royalblue")
    plt.xlabel("Layer index")
    plt.ylabel("Fisher score")
    plt.title(f"Fisher Information per Layer ({MODEL_NAME})")
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, f"fisher_plot_{MODEL_NAME}.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\nFisher 分布图已保存至: {plot_path}")