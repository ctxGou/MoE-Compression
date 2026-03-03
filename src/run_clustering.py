import argparse
import os
import torch
import json
from pathlib import Path
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils import load_fp16_model

def perform_custom_kmeans(merged_outputs, num_clusters, seed, device, max_iters=100):
    """
    对给定的专家输出执行K-Means聚类。
    """
    structured_layers_output = {}

    for layer_idx, experts_data in merged_outputs.items():
        expert_indices = sorted(experts_data.keys())
        num_experts = len(expert_indices)
        if num_experts == 0: continue

        # <--- 修改/新增 1: 将专家输出转为float32以提高计算稳定性 ---
        all_Y = torch.stack([experts_data[i] for i in expert_indices]).to(device).to(torch.float32)
        
        print(f"  - MoE Layer {layer_idx}: Initializing centroids for {num_experts} experts...")
        
        # K-Means++ Initialization
        torch.manual_seed(seed)
        centroids_idx = [torch.randint(0, num_experts, (1,)).item()]
        centroids = all_Y[centroids_idx].clone()

        for k in range(num_clusters - 1):
            dists = torch.stack([
                torch.min(torch.norm(all_Y[i] - centroids, p='fro', dim=(1, 2))**2)
                for i in range(num_experts)
            ])
            
            # <--- 修改/新增 2: 增加nan_to_num_进行数据清洗 ---
            # 清洗dists张量，将可能存在的nan/inf替换为0
            torch.nan_to_num_(dists, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 增加一个极小的epsilon防止分母为0
            eps = torch.finfo(dists.dtype).eps
            probs = dists / (torch.sum(dists) + eps)
            
            if torch.sum(probs).item() == 0:
                remaining_indices = [i for i in range(num_experts) if i not in centroids_idx]
                if not remaining_indices:
                    break
                next_centroid_idx = remaining_indices[torch.randint(0, len(remaining_indices), (1,)).item()]
            else:
                next_centroid_idx = torch.multinomial(probs, 1).item()

            centroids_idx.append(next_centroid_idx)
            centroids = all_Y[centroids_idx].clone()

        # Iterative Assignment and Update (后续部分无需修改)
        print(f"  - MoE Layer {layer_idx}: Starting iterative assignment and update...")
        labels = torch.zeros(num_experts, dtype=torch.long, device=device)
        for i in range(max_iters):
            sim_to_centroids = torch.zeros(num_experts, num_clusters, device=device)
            for expert_j in range(num_experts):
                for cluster_k in range(num_clusters):
                    sim_scores = F.cosine_similarity(all_Y[expert_j], centroids[cluster_k], dim=1)
                    sim_to_centroids[expert_j, cluster_k] = torch.mean(sim_scores)
            
            new_labels = torch.argmax(sim_to_centroids, dim=1)
            
            if torch.equal(labels, new_labels):
                print(f"  - Converged after {i+1} iterations.")
                break
            labels = new_labels

            for cluster_k in range(num_clusters):
                members = all_Y[labels == cluster_k]
                if len(members) > 0:
                    centroids[cluster_k] = torch.mean(members, dim=0)
        else:
            print(f"  - Reached max iterations {max_iters}.")

        # Store results
        layer_cluster_map = defaultdict(list)
        for cluster_id in range(num_clusters):
            experts_in_cluster = [expert_indices[i] for i, label in enumerate(labels) if label == cluster_id]
            layer_cluster_map[f"cluster_{cluster_id}"] = experts_in_cluster
        
        structured_layers_output[f"layer_{layer_idx}"] = dict(layer_cluster_map)
        
        print(f"  - MoE Layer {layer_idx} Clustering Result (k={num_clusters}):")
        for cluster_id, experts in layer_cluster_map.items():
            print(f"    - {cluster_id}: Experts {experts}")
            
    return structured_layers_output

def adaptive_expert_clustering(args):
    """主函数，执行专家聚类流程"""
    

    # 步骤 0: 检查结果文件是否存在，如果存在则跳过
    save_path = Path(args.cluster_results_path)
    save_file = save_path / f"{args.group_method}_groups_{args.num_clusters}.json"
    if save_file.exists():
        print(f"Clustering result file already exists at {save_file}. Skipping.")
        return

    model, tokenizer = load_fp16_model(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = next(model.parameters()).device
    model.eval()


    # 步骤 1: 自动检测模型中的所有MoE层
    print("Step 1: Auto-detecting MoE layers...")
    moe_layer_indices = []
    for i, layer in enumerate(model.model.layers):
        # 兼容 Mixtral, Phi-3.5-MoE 等模型
        is_moe = hasattr(layer, 'block_sparse_moe')
        if not is_moe and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
             is_moe = True
        
        if is_moe:
            moe_layer_indices.append(i)
    
    if not moe_layer_indices:
        print("Error: No MoE layers found in the model.")
        return
    
    print(f"Found {len(moe_layer_indices)} MoE layers at indices: {moe_layer_indices}")

    print("\nStep 2: Collecting expert outputs...")
    expert_outputs = defaultdict(lambda: defaultdict(list))
    
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    dataset = dataset.shuffle(seed=args.seed).select(range(min(args.max_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  - Running forward pass"):
            texts = batch['text']
            inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=args.max_tokens, return_tensors="pt").to(device)
            
            outputs = model(inputs.input_ids, output_hidden_states=True)
            
            # 遍历所有找到的MoE层
            for layer_idx in moe_layer_indices:
                layer = model.model.layers[layer_idx]
                
                # 获取MoE块的输入
                moe_input = layer.input_layernorm(outputs.hidden_states[layer_idx])
                
                # 兼容不同模型的专家模块容器
                experts_container = None
                if hasattr(layer, 'block_sparse_moe'): # For Mixtral, Phi-3.5
                    experts_container = layer.block_sparse_moe.experts
                elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'): # 
                    experts_container = layer.mlp.experts
                
                if experts_container is None:
                    continue

                for expert_idx, expert_mod in enumerate(experts_container):
                    expert_out = expert_mod(moe_input)
                    expert_vector = expert_out.mean(dim=1) # (batch_size, hidden_dim)
                    expert_outputs[layer_idx][expert_idx].append(expert_vector.detach().cpu())

    # 合并所有batch的结果
    merged_outputs = {
        layer_idx: {
            expert_idx: torch.cat(yi_list, dim=0)
            for expert_idx, yi_list in experts.items() if yi_list
        }
        for layer_idx, experts in expert_outputs.items()
    }
    
    print("\nStep 3: Performing K-Means clustering...")
    structured_layers = perform_custom_kmeans(merged_outputs, args.num_clusters, args.seed, device)
    
    results = {
        "args": vars(args),
        "processed_moe_layers": moe_layer_indices, # 记录下实际处理的层
        "cluster_groups": structured_layers
    }
    
    # 确保保存路径存在
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nClustering results saved to: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Expert Clustering for MoE Models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MoE model")
    parser.add_argument("--save_path", type=str, default="./output", help="Base path to save results")
    parser.add_argument("--max_samples", type=int, default=256, help="Number of samples for collecting expert outputs")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of clusters/groups for experts")
    parser.add_argument("--group_method", type=str, default="kmeans", choices=["kmeans", "fixed"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 自动构建路径
    model_name_str = Path(args.model_path).name
    args.cluster_results_path = os.path.join(args.save_path, "cluster_results", model_name_str)
    
    print("--- Starting Expert Clustering ---")
    print(f"Args: {json.dumps(vars(args), indent=2)}")
    adaptive_expert_clustering(args)
    print("--- Expert Clustering Finished ---")