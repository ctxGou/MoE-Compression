# # 文件名: custom_mixtral.py
# # 这个文件定义了一个功能完整的、KV Cache 绝对正常的解码器层

# import torch
# import torch.nn as nn
# from typing import Optional, Tuple

# # 确保能从 transformers 库导入这些必要的组件
# from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm, MixtralSdpaAttention, MixtralSparseMoeBlock

# # 这是我们自定义的、功能完整的 Decoder 层
# class CustomMixtralDecoderLayer(nn.Module):
#     def __init__(self, config, layer_idx):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         # 注意力层，保持不变
#         self.self_attn = MixtralSdpaAttention(config, layer_idx)


#         # MoE 模块，我们先用原始的占位，之后再动态替换它
#         self.block_sparse_moe = MixtralSparseMoeBlock(config)

#         # 归一化层，保持不变
#         self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     # ↓↓↓ 这是本文件的核心！一个从官方源码复制的、100%正确的 forward 方法 ↓↓↓
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         **kwargs,
#     ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         这个 forward 方法的逻辑保证了 KV Cache (接力棒) 能够被正确接收、更新和返回。
#         """
#         # 1. 保存残差输入
#         residual = hidden_states
#         hidden_states = self.input_layernorm(hidden_states)

#         # 2. 调用注意力层，它会处理 KV Cache
#         #    输入: past_key_value (上一轮的缓存)
#         #    输出: present_key_value (本轮更新后的缓存)
#         attn_outputs = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#         )
#         attn_output = attn_outputs[0]
#         self_attn_weights = attn_outputs[1]
#         present_key_value = attn_outputs[2]  # <--- 这就是更新后的“接力棒”！

#         # 3. 第一个残差连接
#         hidden_states = residual + attn_output

#         # 4. MoE 模块前的准备
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)

#         # 5. 调用 MoE 模块
#         #    它的返回值是一个元组 (moe_output, router_logits)
#         moe_outputs = self.block_sparse_moe(hidden_states)
#         hidden_states = moe_outputs[0]

#         # 6. 第二个残差连接
#         hidden_states = residual + hidden_states

#         # 7. 准备返回值，这是最关键的一步
#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         # 如果需要使用缓存，就把更新后的“接力棒” present_key_value 返回！
#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs