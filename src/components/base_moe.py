# /home/Anonymous/TD-MoE/src/components/base_moe.py
    
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import gc
# from config import get_num_experts


# class TuckerDecomposedMLP(nn.Module):
#     """
#     去白化 -> 分解计算 -> 去白化" 流程。
#     """
#     def __init__(self, core, U_exp, U_out, U_in, S2, S3, device, dtype):
#         super().__init__()
#         self.register_buffer('core', core.to(device=device, dtype=dtype))
#         self.register_buffer('U_exp', U_exp.to(device=device, dtype=dtype))
#         self.register_buffer('U_out', U_out.to(device=device, dtype=dtype))
#         self.register_buffer('U_in', U_in.to(device=device, dtype=dtype))
#         self.register_buffer('S2', S2.to(device=device, dtype=dtype) if S2 is not None else None)
#         self.register_buffer('S3', S3.to(device=device, dtype=dtype) if S3 is not None else None)
#         self.bias = None

#     def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
#         original_dtype = x.dtype
#         x_f32 = x.to(torch.float32)

#         # 1. 输入端去白化和输入投影
#         x_transformed = x_f32 @ self.S2.to(torch.float32) if self.S2 is not None else x_f32        
#         x_in_proj = x_transformed @ self.U_in.to(torch.float32) # Shape: (num_tokens, rank_in)

#         U_exp_selected = self.U_exp[expert_indices].to(torch.float32) 
#         core_tensor = self.core.to(torch.float32) 
        
#         num_tokens, rank_expert = U_exp_selected.shape
#         _, rank_out, rank_in = core_tensor.shape # 注意: core shape 是 (r_expert, r_out, r_in)

#         output_from_core = torch.zeros(num_tokens, rank_out, device=x.device, dtype=torch.float32)
        
#         for r in range(rank_expert):
#             U_exp_r = U_exp_selected[:, r] 
#             core_r = core_tensor[r, :, :] # Shape: (rank_out, rank_in)
            
#             term1 = x_in_proj * U_exp_r.unsqueeze(1) # Shape: (num_tokens, rank_in)
        
#             term2 = term1 @ core_r.T
            
#             output_from_core += term2

#         # 3. 输出投影和输出端去白化
#         output_proj = output_from_core @ self.U_out.T.to(torch.float32)
#         final_output = output_proj @ self.S3.to(torch.float32) if self.S3 is not None else output_proj  

#         if self.bias is not None:
#             final_output = final_output + self.bias.to(torch.float32)
        
#         return final_output.to(original_dtype)



# class TuckerDecomposedMLP(nn.Module):
#     def __init__(self, core, U_exp, U_out, U_in, S2, S3, d_out, device, dtype):
#         super().__init__()
        
#         # <--- 调试功能：在初始化时检查输入张量的数值稳定性
#         print(f"--- Initializing TuckerDecomposedMLP for d_out={d_out} ---")
#         self._check_numerical_stability(core=core, U_exp=U_exp, U_out=U_out, U_in=U_in, S2=S2, S3=S3)
        
#         self.register_buffer('core', core.to(device=device, dtype=dtype))
#         self.register_buffer('U_exp', U_exp.to(device=device, dtype=dtype))
#         self.register_buffer('U_out', U_out.to(device=device, dtype=dtype))
#         self.register_buffer('U_in', U_in.to(device=device, dtype=dtype))
        
#         self.register_buffer('S2_inv', torch.linalg.inv(S2).to(device=device, dtype=dtype) if S2 is not None else None)
#         self.register_buffer('S3', S3.to(device=device, dtype=dtype) if S3 is not None else None)
        
#         self.d_out = d_out
#         self.bias = None
#         self.debug_mode = False

#     def _check_numerical_stability(self, **kwargs):
#         """在初始化时检查所有输入张量，防止坏数据进入模型"""
#         for name, tensor in kwargs.items():
#             if tensor is not None:
#                 if torch.isnan(tensor).any() or torch.isinf(tensor).any():
#                     raise ValueError(f"CRITICAL ERROR: Tensor '{name}' contains NaN or Inf during initialization!")
#                 tensor_max_abs = torch.max(torch.abs(tensor))
#                 if tensor_max_abs > 1e4:
#                      print(f"DEBUG WARNING: Tensor '{name}' has large values (max abs: {tensor_max_abs:.2e})")

#     def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
#         x_f32 = x.to(torch.float32)
#         batch_size = x_f32.shape[0]
#         final_output = torch.zeros(batch_size, self.d_out, device=x.device, dtype=torch.float32)
        
#         if self.debug_mode and torch.isnan(x_f32).any():
#             print("DEBUG CRITICAL: Input 'x' to forward pass contains NaN!")
#             # 返回零以避免崩溃，但在日志中这是一个严重警告
#             return final_output

#         unique_experts = torch.unique(expert_indices)
        
#         for expert_id in unique_experts:
#             token_mask = (expert_indices == expert_id)
#             tokens_for_expert = x_f32[token_mask]
            
#             if tokens_for_expert.shape[0] == 0: continue

#             # --- 步骤 1: 输入去白化 ---            
#             x_dewhitened_space = tokens_for_expert @ self.S2_inv.T.to(torch.float32) if self.S2_inv is not None else tokens_for_expert
            
#             if self.debug_mode and (torch.isnan(x_dewhitened_space).any() or torch.isinf(x_dewhitened_space).any()):
#                 print(f"DEBUG (expert {expert_id.item()}): NaN/Inf detected AFTER input de-whitening!")
#                 continue # 跳过这个坏的expert

#             x_in_proj = x_dewhitened_space @ self.U_in.to(torch.float32)
#             if self.debug_mode and (torch.isnan(x_in_proj).any() or torch.isinf(x_in_proj).any()):
#                 print(f"DEBUG (expert {expert_id.item()}): NaN/Inf detected AFTER input projection!")
#                 continue

#             # --- 步骤 2: 重建核心 ---
#             U_exp_e = self.U_exp[expert_id].to(torch.float32)
#             core_e = torch.einsum('e, eoi -> oi', U_exp_e, self.core.to(torch.float32))
#             if self.debug_mode and (torch.isnan(core_e).any() or torch.isinf(core_e).any()):
#                 print(f"DEBUG (expert {expert_id.item()}): NaN/Inf detected in reconstructed core!")
#                 continue

#             # --- 步骤 3: 应用核心和输出投影 ---
#             output_from_core = x_in_proj @ core_e.T
#             output_proj = output_from_core @ self.U_out.T.to(torch.float32)
#             if self.debug_mode and (torch.isnan(output_proj).any() or torch.isinf(output_proj).any()):
#                 print(f"DEBUG (expert {expert_id.item()}): NaN/Inf detected AFTER output projection!")
#                 continue

#             # --- 步骤 4: 输出去白化 ---
#             expert_output = output_proj @ self.S3.T.to(torch.float32) if self.S3 is not None else output_proj
#             if self.debug_mode and (torch.isnan(expert_output).any() or torch.isinf(expert_output).any()):
#                 print(f"DEBUG (expert {expert_id.item()}): NaN/Inf detected AFTER output de-whitening!")
#                 continue
            
#             final_output[token_mask] = expert_output

#         if self.bias is not None:
#             final_output = final_output + self.bias.to(torch.float32)
        
#         if self.debug_mode and (torch.isnan(final_output).any() or torch.isinf(final_output).any()):
#             print("DEBUG CRITICAL: Final output contains NaN/Inf before returning! Replacing with zeros.")
#             final_output = torch.nan_to_num(final_output, nan=0.0, posinf=1.0, neginf=-1.0)
            
#         # return final_output
#         return final_output

def cond_check_tensor(tensor, name):
    """
    Print the condition number of the given tensor.
    2D tensors are trivial, 3D tensors report a list of cond for each slice along the first dimension.
    """
    if tensor is None:
        return

    def _safe_cond(mat: torch.Tensor):
        probe = mat
        if probe.dtype in (torch.float16, torch.bfloat16):
            probe = probe.to(torch.float32)
        try:
            return torch.linalg.cond(probe)
        except NotImplementedError:
            # Fallback for backends/dtypes without CUDA SVD support
            return torch.linalg.cond(probe.cpu().to(torch.float32))
        except RuntimeError as e:
            print(f"  - Skipping cond({name}) due to runtime error: {e}")
            return None

    if tensor.ndim == 2:
        cond_number = _safe_cond(tensor)
        if cond_number is not None:
            print(f"  - Condition number of {name}: {float(cond_number):.2e}")
    elif tensor.ndim == 3:
        cond_numbers = []
        for i in range(tensor.shape[0]):
            slice_i = tensor[i, :, :]
            cond_i = _safe_cond(slice_i)
            if cond_i is not None:
                cond_numbers.append(float(cond_i))
        if cond_numbers:
            print(f"  - Condition numbers of {name} slices: {[f'{c:.2e}' for c in cond_numbers]}")
    else:
        return

class TuckerDecomposedMLP(nn.Module):
    """
    融合版本：恢复使用基于秩循环的快速计算逻辑。
    由上层的 BaseTuckerDecomposedMoE 保证传入的 x (chunk) 不会过大。
    """
    # __init__ 和 _check_numerical_stability 方法保持不变
    def __init__(self, core, U_exp, U_out, U_in, S2, S3, d_out, device, dtype):
        super().__init__()
        print(f"--- Initializing TuckerDecomposedMLP for d_out={d_out} ---")
        self._check_numerical_stability(core=core, U_exp=U_exp, U_out=U_out, U_in=U_in, S2=S2, S3=S3)
        
        self.register_buffer('core', core.to(device=device, dtype=dtype))
        self.register_buffer('U_exp', U_exp.to(device=device, dtype=dtype))
        self.register_buffer('U_out', U_out.to(device=device, dtype=dtype))
        self.register_buffer('U_in', U_in.to(device=device, dtype=dtype))
        
        self.register_buffer('S2_inv', torch.linalg.inv(S2).to(device=device, dtype=dtype) if S2 is not None else None)
        self.register_buffer('S3', S3.to(device=device, dtype=dtype) if S3 is not None else None)
        
        cond_check_tensor(self.core, "core")
        cond_check_tensor(self.U_exp, "U_exp")
        cond_check_tensor(self.U_out, "U_out")
        cond_check_tensor(self.U_in, "U_in")
        cond_check_tensor(self.S2_inv, "S2_inv")
        cond_check_tensor(self.S3, "S3")

        self.d_out = d_out
        self.bias = None

    def _check_numerical_stability(self, **kwargs):
        for name, tensor in kwargs.items():
            if tensor is not None:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ValueError(f"CRITICAL ERROR: Tensor '{name}' contains NaN or Inf during initialization!")

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)

        # 步骤 1: 输入去白化和输入投影 (对整个chunk进行)
        x_dewhitened = x_f32 @ self.S2_inv.T.to(torch.float32) if self.S2_inv is not None else x_f32
        x_in_proj = x_dewhitened @ self.U_in.to(torch.float32)

        # 步骤 2: 选择专家因子 (对整个chunk进行)
        U_exp_selected = self.U_exp[expert_indices].to(torch.float32)
        core_tensor = self.core.to(torch.float32)
        
        num_tokens, _ = x_in_proj.shape
        rank_expert, rank_out, _ = core_tensor.shape

        # 步骤 3: 恢复高效的秩循环计算逻辑
        output_from_core = torch.zeros(num_tokens, rank_out, device=x.device, dtype=torch.float32)
        
        for r in range(rank_expert):
            U_exp_r = U_exp_selected[:, r].unsqueeze(1)
            core_r = core_tensor[r, :, :]
            term = (x_in_proj @ core_r.T) * U_exp_r
            output_from_core += term
        
        # 步骤 4: 输出投影和输出去白化
        output_proj = output_from_core @ self.U_out.T.to(torch.float32)
        final_output = output_proj @ self.S3.T.to(torch.float32) if self.S3 is not None else output_proj

        if self.bias is not None:
            final_output = final_output + self.bias.to(torch.float32)
            
        return final_output
        


class BaseTuckerDecomposedMoE(nn.Module, ABC):
    def __init__(self, config, original_moe, decompose_data, layer_idx, cluster_type, cluster_info, model_dtype):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        try:
            self.num_experts = config.num_local_experts
        except:
            self.num_experts = config.n_routed_experts
            
        # self.num_experts = get_num_experts(config)
        self.device = next(original_moe.parameters()).device
        self.dtype = model_dtype
        self.cluster_type = cluster_type
        self.cluster_info = cluster_info
        # self.chunk_size = 4096 

        self._initialize_model_specific_modules(original_moe)
        
        # self.tucker_experts, expert_to_cluster_map_cpu = self._build_tucker_experts(decompose_data, layer_idx)
        
        # _build_tucker_experts 现在可能返新增返回 expert_global_to_local_map_cpu
        results = self._build_tucker_experts(decompose_data, layer_idx)
        self.tucker_experts, expert_to_cluster_map_cpu, expert_global_to_local_map_cpu = results
        print(f"DEBUG[base_moe.py]: {expert_global_to_local_map_cpu=}")
        
        self.register_buffer('expert_to_cluster_map_buffer', expert_to_cluster_map_cpu)
        
        # 新增一个映射表，用于将全局专家ID转换为分组内的局部ID
        self.register_buffer('expert_global_to_local_map_buffer', expert_global_to_local_map_cpu)
                
        
        self._load_biases(original_moe)

    @abstractmethod
    def _initialize_model_specific_modules(self, original_moe): pass
    @abstractmethod
    def _route_tokens(self, hidden_states): pass
    # @abstractmethod
    # def _format_output(self, final_hidden_states, router_logits): pass
    @abstractmethod
    def _build_tucker_experts(self, decompose_data, layer_idx): pass
    @abstractmethod
    def _load_biases(self, original_moe): pass

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        selected_experts, routing_weights, router_logits = self._route_tokens(hidden_states_flat)
        
        # final_hidden_states = torch.zeros_like(hidden_states_flat)
        final_hidden_states = torch.zeros_like(hidden_states_flat, dtype=torch.float32)


        for k in range(self.num_experts_per_tok):
            k_experts = selected_experts[:, k]
            k_weights = routing_weights[:, k]
            
            k_clusters = self.expert_to_cluster_map_buffer.to(k_experts.device)[k_experts]

            for cluster_id in torch.unique(k_clusters):
                token_indices = torch.where(k_clusters == cluster_id)[0]
                
                if token_indices.numel() == 0:
                    continue

                current_states = hidden_states_flat[token_indices]
                original_expert_ids = k_experts[token_indices]
                
                # 决定使用哪个 cluster_key 来访问 tucker_experts
                if self.cluster_type == 'group':
                    # 根据cluster_id找到对应的组名
                    cluster_key = list(self.cluster_info.keys())[cluster_id.item()]
                else: # global 模式
                    cluster_key = f'cluster_{cluster_id.item()}'                
                                
                
                cluster_expert_modules = self.tucker_experts[cluster_key]
                # cluster_expert_modules = self.tucker_experts[f'cluster_{cluster_id.item()}']
                
                # 将局部专家ID传递给MLP模块
                local_expert_ids = self.expert_global_to_local_map_buffer.to(original_expert_ids.device)[original_expert_ids]
                
                gate_output = cluster_expert_modules.gate_proj(current_states, local_expert_ids)
                up_output = cluster_expert_modules.up_proj(current_states, local_expert_ids)
                intermediate_states = self.act_fn(gate_output) * up_output
                down_output = cluster_expert_modules.down_proj(intermediate_states, local_expert_ids)                
                
                
                # gate_output = cluster_expert_modules.gate_proj(current_states, original_expert_ids)
                # up_output = cluster_expert_modules.up_proj(current_states, original_expert_ids)
                # intermediate_states = self.act_fn(gate_output) * up_output
                # down_output = cluster_expert_modules.down_proj(intermediate_states, original_expert_ids)
                
                weighted_output = down_output * k_weights[token_indices].unsqueeze(1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        


        return final_hidden_states.to(hidden_states.dtype), router_logits
        # return final_hidden_states.to(hidden_states.dtype)
