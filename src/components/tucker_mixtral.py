# /home/Anonymous/TD-MoE/src/components/tucker_mixtral.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_moe import BaseTuckerDecomposedMoE, TuckerDecomposedMLP

class MixtralTuckerDecomposedMoE(BaseTuckerDecomposedMoE):
    """针对 Mixtral 模型的 Tucker-Decomposed MoE 实现"""

    def _initialize_model_specific_modules(self, original_moe):
        # 复制原始模型的门控层和激活函数
        self.gate = original_moe.gate
        # Mixtral的激活函数在每个expert子模块里
        self.act_fn = original_moe.experts[0].act_fn
        
        # 在在子类 MixtralTuckerDecomposedMoE 中，利用这个 self.dtype，通过 self.gate.to(dtype=self.dtype)，把可能被转为 float32 的 gate 模块强制转换回 float16
        self.gate = self.gate.to(dtype=self.dtype)
        # print(f"{self.dtype=}")

        
    def _route_tokens(self, hidden_states):
        # Mixtral 的路由逻辑
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # routing_weights = routing_weights.to(hidden_states.dtype)
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) 
        return selected_experts, routing_weights, router_logits

    # def _format_output(self, final_hidden_states, router_logits):
    #     return final_hidden_states, router_logits

    def _build_tucker_experts(self, decompose_data, layer_idx):
        tucker_experts = nn.ModuleDict()
        
        # 对于 'global' 模式，所有专家共享一套分解参数，视为一个聚类
        if self.cluster_type == 'global':
            print(f"  (层 {layer_idx}) 正在构建全局的 Tucker experts...")
            
            # decompose_data 的 key 是 'w1', 'w3', 'w2'
            gate_proj_data = decompose_data['w1']
            up_proj_data = decompose_data['w3']
            down_proj_data = decompose_data['w2']            
                
            
            # 提取 gate_proj (w1) 的组件
            gate_core = gate_proj_data['core']
            gate_U_exp, gate_U_out, gate_U_in = gate_proj_data['factors']
            gate_S2 = gate_proj_data['Cholesky']['S2']
            gate_S3 = gate_proj_data['Cholesky']['S3']
            
            # print(f"{ gate_proj_data['factors']=}")
            # print(f"{gate_S2=}")
            # print(f"{gate_S3=}")
            
            
            # 提取 up_proj (w3) 的组件
            up_core = up_proj_data['core']
            up_U_exp, up_U_out, up_U_in = up_proj_data['factors']
            up_S2 = up_proj_data['Cholesky']['S2']
            up_S3 = up_proj_data['Cholesky']['S3']

            # 提取 down_proj (w2) 的组件
            down_core = down_proj_data['core']
            down_U_exp, down_U_out, down_U_in = down_proj_data['factors']
            down_S2 = down_proj_data['Cholesky']['S2']
            down_S3 = down_proj_data['Cholesky']['S3']

            # 实例化三个 TuckerDecomposedMLP 模块
            gate_proj_mlp = TuckerDecomposedMLP(gate_core, gate_U_exp, gate_U_out, gate_U_in, gate_S2, gate_S3, d_out=self.config.intermediate_size, device = self.device, dtype = self.dtype)
            up_proj_mlp = TuckerDecomposedMLP(up_core, up_U_exp, up_U_out, up_U_in, up_S2, up_S3, d_out=self.config.intermediate_size, device = self.device, dtype = self.dtype)
            down_proj_mlp = TuckerDecomposedMLP(down_core, down_U_exp, down_U_out, down_U_in, down_S2, down_S3, d_out=self.config.hidden_size, device = self.device, dtype = self.dtype)

            # 将它们放入一个 ModuleDict，代表 "cluster_0"
            tucker_experts['cluster_0'] = nn.ModuleDict({
                'gate_proj': gate_proj_mlp, # 对应 w1
                'up_proj': up_proj_mlp,     # 对应 w3
                'down_proj': down_proj_mlp  # 对应 w2
            })
            
            # 在 global 模式下，所有专家都映射到聚类0
            expert_to_cluster_map = torch.zeros(self.num_experts, dtype=torch.long)
            expert_global_to_local_map = torch.arange(self.num_experts, dtype=torch.long)

        # 'group' 模式
        elif self.cluster_type == 'group':
            if self.cluster_info is None:
                raise ValueError("`cluster_info` 是 'group' 模式所必需的。")
            
            print(f"  (层 {layer_idx}) 正在构建分组的 Tucker experts...")
            
            expert_to_cluster_map = torch.zeros(self.num_experts, dtype=torch.long)
            expert_global_to_local_map = torch.zeros(self.num_experts, dtype=torch.long)
            
            group_name_to_id = {name: i for i, name in enumerate(self.cluster_info.keys())}

            for group_name, expert_indices in self.cluster_info.items():
                print(f"    - 正在为 {group_name} (专家: {expert_indices}) 构建模块")
                
                cluster_id = group_name_to_id[group_name]
                expert_to_cluster_map[expert_indices] = cluster_id
                
                for local_idx, global_idx in enumerate(expert_indices):
                    expert_global_to_local_map[global_idx] = local_idx

                # --- 数据访问逻辑现在安全地位于 'group' 分支的循环内 ---
                group_decomp_data = decompose_data[group_name]
                gate_proj_data = group_decomp_data['w1']
                up_proj_data = group_decomp_data['w3']
                down_proj_data = group_decomp_data['w2']

                gate_core, (gate_U_exp, gate_U_out, gate_U_in), gate_chol = gate_proj_data['core'], gate_proj_data['factors'], gate_proj_data['Cholesky']
                up_core, (up_U_exp, up_U_out, up_U_in), up_chol = up_proj_data['core'], up_proj_data['factors'], up_proj_data['Cholesky']
                down_core, (down_U_exp, down_U_out, down_U_in), down_chol = down_proj_data['core'], down_proj_data['factors'], down_proj_data['Cholesky']

                gate_proj_mlp = TuckerDecomposedMLP(gate_core, gate_U_exp, gate_U_out, gate_U_in, gate_chol.get('S2'), gate_chol.get('S3'), self.config.intermediate_size, self.device, self.dtype)
                up_proj_mlp = TuckerDecomposedMLP(up_core, up_U_exp, up_U_out, up_U_in, up_chol.get('S2'), up_chol.get('S3'), self.config.intermediate_size, self.device, self.dtype)
                down_proj_mlp = TuckerDecomposedMLP(down_core, down_U_exp, down_U_out, down_U_in, down_chol.get('S2'), down_chol.get('S3'), self.config.hidden_size, self.device, self.dtype)

                tucker_experts[f'cluster_{cluster_id}'] = nn.ModuleDict({
                    'gate_proj': gate_proj_mlp,
                    'up_proj': up_proj_mlp,
                    'down_proj': down_proj_mlp
                })
        
        else:
            raise ValueError(f"Unknown decomposition type: {self.cluster_type}")
            
        return tucker_experts, expert_to_cluster_map, expert_global_to_local_map

    def _load_biases(self, original_moe):
        # Mixtral的FFN层没有偏置项，直接跳过
        pass