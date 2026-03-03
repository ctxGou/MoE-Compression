# src/config.py
from dataclasses import dataclass
from typing import Dict, Type
import torch
import torch.nn as nn
from components.tucker_mixtral import MixtralTuckerDecomposedMoE
from components.tucker_phi import PhiTuckerDecomposedMoE
# from components.tucker_deepseek import DeepseekTuckerDecomposedMoE

# ---------------------------------------------------------------------------- #
#                     FFN Role & Model Configuration                           #
# ---------------------------------------------------------------------------- #

# 定义FFN中每个“角色”的通用行为模式
FFN_ROLE_CONFIGS = {
    'gate': {
        'd_in_key': 'hidden_size',       # 输入总是模型的隐藏维度
        'd_out_key': 'intermediate_size', # 输出总是模型的中间维度
        'hook_target': 'self'             # 钩子挂在MoE层本身，捕获其输入
    },
    'up': {
        'd_in_key': 'hidden_size',
        'd_out_key': 'intermediate_size',
        'hook_target': 'self'
    },
    'down': {
        'd_in_key': 'intermediate_size',  # 输入是模型的中间维度
        'd_out_key': 'hidden_size',       # 输出是模型的隐藏维度
        'hook_target': 'down_proj'        # 钩子需挂在down_proj自己的输入端
    }
}

@dataclass
class MoEModelConfig:
    """模型配置的数据类，结构清晰"""
    model_name_pattern: str
    moe_layer_pattern: str
    layer_class_name: str
    is_stacked: bool
    role_map: Dict[str, str]  # e.g., {'gate': 'w1', 'up': 'w3', 'down': 'w2'}
    decomposed_class: Type[nn.Module]

# --- 模型注册表 ---
# 所有模型特有的信息都集中在此，主程序逻辑保持通用
MODEL_REGISTRY = {
    'mixtral': MoEModelConfig(
        model_name_pattern='Mixtral',
        moe_layer_pattern='model.layers.{}.block_sparse_moe',
        layer_class_name='MixtralDecoderLayer',
        is_stacked=False,
        role_map={'gate': 'w1', 'up': 'w3', 'down': 'w2'},
        decomposed_class=MixtralTuckerDecomposedMoE
    ),
    # 'deepseek': MoEModelConfig(
    #     model_name_pattern='deepseek-moe',
    #     moe_layer_pattern='model.layers.{}.mlp',
    #     layer_class_name='DeepseekDecoderLayer',
    #     is_stacked=False,
    #     role_map={'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'},
    #     decomposed_class=DeepseekTuckerDecomposedMoE
    #     # decomposed_class=DeepseekTuckerDecomposedMoE
    # ),
    'phi35': MoEModelConfig(
        model_name_pattern='Phi-3.5-MoE-instruct',
        moe_layer_pattern='model.layers.{}.block_sparse_moe',   # 'model.layers.0.block_sparse_moe.experts.13.w2.weight'
        layer_class_name='PhiMoEDecoderLayer',
        is_stacked=False,
        role_map={'gate': 'w1', 'up': 'w3', 'down': 'w2'}, 
        decomposed_class=PhiTuckerDecomposedMoE 
    ),    
}

# ---------------------------------------------------------------------------- #
#                               Helper Functions                               #
# ---------------------------------------------------------------------------- #

def get_model_config(model_name_or_path: str) -> MoEModelConfig:
    """根据模型路径找到对应的配置实例"""
    for _, config in MODEL_REGISTRY.items():
        if config.model_name_pattern in model_name_or_path:
            return config
    raise ValueError(f"Model config not found for {model_name_or_path}. Please add it to MODEL_REGISTRY in src/config.py")

def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """根据层名字符串获取模型的具体模块"""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        try:
            module = getattr(module, part)
        except AttributeError:
            raise AttributeError(f"Module {layer_name} not found in model.")
    return module

def set_layer_by_name(model: nn.Module, layer_name: str, new_layer: nn.Module):
    """根据层名字符串替换模型的具体模块"""
    parts = layer_name.split('.')
    parent_module = model
    for part in parts[:-1]:
        try:
            parent_module = getattr(parent_module, part)
        except AttributeError:
            raise AttributeError(f"Parent module for {layer_name} not found.")
    setattr(parent_module, parts[-1], new_layer)
    
    
def get_num_experts(config) -> int:
    """
    从模型配置中安全地获取专家总数，兼容多种命名方式。
    """
    expert_keys_to_try = ['num_local_experts', 'num_experts', 'n_routed_experts']
    num_experts = None
    found_key = None

    for key in expert_keys_to_try:
        num_experts = getattr(config, key, None)
        if num_experts is not None:
            found_key = key
            break

    if num_experts is None:
        raise AttributeError(f"无法从模型配置中确定专家数量。已尝试 {expert_keys_to_try}。")
    
    print(f"(Helper) 通过 '{found_key}' 获取专家数量: {num_experts}")
    return num_experts    