# file: components/loaded_tucker_moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from .base_moe import BaseTuckerDecomposedMoE, TuckerDecomposedMLP

# class TuckerDecomposedMLP(nn.Module):
#     """
#     一个模块，用于封装单个 FFN 层的 Tucker 分解组件 (如 w1, w2, 或 w3)。
#     它从外部接收预先计算好的张量，并将它们注册为缓冲区。
#     它的 forward 方法接收 token 的隐状态和它们被分配到的专家索引，
#     然后执行高效的分解计算。
#     """
#     def __init__(self, core, U_exp, U_out, U_in, S2, S3, device, dtype):
#         super().__init__()
#         # 将所有预分解的张量注册为模型的缓冲区（buffer）。
#         # 缓冲区是模型状态的一部分，但不会被视为可训练参数。
#         self.register_buffer('core', core.to(device=device, dtype=dtype))
#         self.register_buffer('U_exp', U_exp.to(device=device, dtype=dtype))
#         # U_out 和 U_in 来自于线性层分解，它们本身就是权重矩阵
#         self.register_buffer('U_out', U_out.to(device=device, dtype=dtype))
#         self.register_buffer('U_in', U_in.to(device=device, dtype=dtype))
#         # S2 和 S3 是用于白化/去白化的 Cholesky 矩阵
#         self.register_buffer('S2', S2.to(device=device, dtype=dtype) if S2 is not None else None)
#         self.register_buffer('S3', S3.to(device=device, dtype=dtype) if S3 is not None else None)

#     def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
#         original_dtype = x.dtype
#         x_f32 = x.to(torch.float32)

#         # 1. 输入端去白化和输入投影
#         x_dewhitened = x_f32 @ self.S2.to(torch.float32) if self.S2 is not None else x_f32
#         x_in_proj = x_dewhitened @ self.U_in.to(torch.float32)

#         # 2. 根据 expert_indices 选择对应的专家因子
#         U_exp_selected = self.U_exp[expert_indices].to(torch.float32)
#         core_tensor = self.core.to(torch.float32)

#         # 3. 将一步 einsum 分解为两步，以控制内存使用
#         # 第一步：为每个 token 创建一个专属的核心矩阵
#         # 'te, eoi -> toi'
#         # t: token, e: expert_rank, o: out_rank, i: in_rank
#         per_token_core = torch.einsum('te, eoi -> toi', U_exp_selected, core_tensor)

#         # 第二步：将专属核心矩阵应用到输入投影上
#         # 'toi, ti -> to'
#         output_from_core = torch.einsum('toi, ti -> to', per_token_core, x_in_proj)

#         # 4. 输出投影和输出端白化
#         output_proj = output_from_core @ self.U_out.T.to(torch.float32)
#         final_output = output_proj @ self.S3.T.to(torch.float32) if self.S3 is not None else output_proj

#         return final_output.to(original_dtype)
    

class PhiTuckerDecomposedMoE(nn.Module):
    """
    此类用于替换原始模型中的 MoE 模块 (例如 block_sparse_moe)。
    它从文件中加载预先计算好的 Tucker 分解数据来构建专家网络，
    并复制原始模型的门控（gating）层。
    """
    def __init__(self, config, original_moe, decompose_data, layer_idx, cluster_type, cluster_info, model_dtype):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.device = next(original_moe.parameters()).device
        self.dtype = model_dtype
        self.cluster_info = cluster_info

        # 1. 复制原始 MoE 模块的非专家部分（门控和激活函数）
        self._initialize_model_specific_modules(original_moe)

        # 2. 从加载的 `decompose_data` 构建所有专家的 FFN 计算模块
        self.experts_ffn = self._build_experts_from_data(decompose_data)

    def _initialize_model_specific_modules(self, original_moe):
        """复制门控层和激活函数。"""
        # 门控（路由）权重是独立于专家的，直接复制
        self.gate = original_moe.gate
        # 激活函数也直接从原始专家中获取
        self.act_fn = original_moe.experts[0].act_fn
        # 确保 gate 的数据类型与模型一致
        self.gate = self.gate.to(dtype=self.dtype)

    def _build_experts_from_data(self, decompose_data):
        """
        遍历加载的分解数据，为每个 FFN 层 (w1, w2, w3) 创建一个 TuckerDecomposedMLP 实例。
        """
        experts_ffn = nn.ModuleDict()
        
        # decompose_data 的 key 是 'w1', 'w3', 'w2'
        for role_name, decomp_data_for_role in decompose_data.items():
            core = decomp_data_for_role['core']
            # factors 是一个包含 U_exp, U_out, U_in 的列表或元组
            U_exp, U_out, U_in = decomp_data_for_role['factors']
            # Cholesky 包含 S2 (输入白化) 和 S3 (输出白化)
            S2 = decomp_data_for_role['Cholesky']['S2']
            S3 = decomp_data_for_role['Cholesky']['S3']

            # 创建并存储一个处理该 FFN 层的分解模块
            experts_ffn[role_name] = TuckerDecomposedMLP(
                core, U_exp, U_out, U_in, S2, S3, self.config.intermediate_size, self.device, self.dtype
            )
        return experts_ffn

    def _route_tokens(self, hidden_states):
        """执行路由计算，确定每个 token 由哪些专家处理。"""
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        return selected_experts, routing_weights, router_logits

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. 获取路由结果
        selected_experts, routing_weights, router_logits = self._route_tokens(hidden_states_flat)
        
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # 将 token 分组，以便进行高效的批处理
        # expert_mask tensor, shape is (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 2. 循环遍历每个专家，处理分配给它的所有 token
        for expert_idx in range(self.num_experts):
            # top_x: a tensor of indices of tokens routed to this expert
            # idx: a tensor of indices indicating whether it's the token's top-1 or top-2 expert
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue
                
            # 提取这些 token 的隐状态和对应的路由权重
            current_states = hidden_states_flat[top_x]
            current_routing_weights = routing_weights[top_x, idx].unsqueeze(1)
            
            # 创建一个包含专家索引的张量，传递给 TuckerDecomposedMLP
            # 因为这里所有 token 都属于同一个 expert_idx，所以张量所有值都一样
            expert_indices_for_tokens = torch.full_like(top_x, fill_value=expert_idx)

            # 3. 执行 FFN 计算流程
            # a. Gate (w1) 和 Up (w3) 路径
            gate_output = self.experts_ffn.w1(current_states, expert_indices_for_tokens)
            up_output = self.experts_ffn.w3(current_states, expert_indices_for_tokens)
            
            # b. 激活和相乘
            intermediate_states = self.act_fn(gate_output) * up_output
            
            # c. Down (w2) 路径
            down_output = self.experts_ffn.w2(intermediate_states, expert_indices_for_tokens)
            
            # 4. 应用路由权重并将结果加回到最终输出中
            weighted_output = down_output * current_routing_weights
            
            if weighted_output.dtype != final_hidden_states.dtype:
                weighted_output = weighted_output.to(dtype=final_hidden_states.dtype)
            if weighted_output.device != final_hidden_states.device:
                weighted_output = weighted_output.to(device=final_hidden_states.device)
                            
            final_hidden_states.index_add_(0, top_x, weighted_output)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits
    