import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers.models.mixtral.modeling_mixtral import *
# from component.svd_mixtral_sharing import SVD_MixtralSparseMoeBlock
from tqdm import tqdm
import pandas as pd


from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig

# def load_quant_model(model_path: str):
#     """加载 4-bit 量化模型，用于协方差计算"""
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         quantization_config=quant_config,
#         device_map="auto",
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model.eval()
#     return model, tokenizer

def load_cpu_model(model_path: str):
    """ cpu 加载原始 FP16 模型"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # print(f"{model=}")
    model.eval()
    return model, tokenizer


def load_fp16_model(model_path: str):
    """加载原始 FP16 模型，用于白化和 Tucker 分解"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    # model.config.use_cache=True
    model.config.use_cache = False
    return model, tokenizer

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


# def get_model_from_local_gpu(model_id, model_name, mode='custom'):
#     from accelerate import init_empty_weights, load_checkpoint_and_dispatch
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     if mode == 'custom':
#         # 加载自定义模型
#         print(f"get model from local gpu Loading model from {model_id}")
#         pruned_dict = torch.load(model_id, map_location='cpu')
#         tokenizer = pruned_dict['tokenizer']
#         model = pruned_dict['model']
        
#         # 使用 accelerate 的 load_checkpoint_and_dispatch 加载和分配模型
#         model = load_checkpoint_and_dispatch(
#             model=model,
#             checkpoint=model_id,
#             device_map="auto",
#             no_split_module_classes=['MixtralDecoderLayer','DeepseekDecoderLayer','PhiMoEDecoderLayer']
#         )
#         del pruned_dict
#         gc.collect()
#         torch.cuda.empty_cache()
#         return model, tokenizer

#     elif mode == 'huggingface':
#         # 加载 Huggingface 模型和 tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # 初始化空模型
#         # with init_empty_weights():
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         print(model_id)
#         # 使用 accelerate 的 load_checkpoint_and_dispatch 加载和分配模型
#         model = load_checkpoint_and_dispatch(
#             model=model,
#             checkpoint=model_id,
#             device_map='auto',
#             no_split_module_classes=['MixtralDecoderLayer','DeepseekDecoderLayer','PhiMoEDecoderLayer']
#         )

#         return model, tokenizer

#     else:
#         raise ValueError("Invalid mode. Choose either 'custom' or 'huggingface'.")



# class MoEGate(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.top_k = config.num_experts_per_tok
#         self.n_routed_experts = config.n_routed_experts

#         self.scoring_func = config.scoring_func
#         self.alpha = config.aux_loss_alpha
#         self.seq_aux = config.seq_aux

#         # topk selection algorithm
#         self.norm_topk_prob = config.norm_topk_prob
#         self.gating_dim = config.hidden_size
#         self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         import torch.nn.init  as init
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
#     def forward(self, hidden_states):
#         bsz, seq_len, h = hidden_states.shape        
#         ### compute gating score
#         hidden_states = hidden_states.view(-1, h)
#         logits = F.linear(hidden_states, self.weight, None)
#         if self.scoring_func == 'softmax':
#             scores = logits.softmax(dim=-1)
#         else:
#             raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
#         ### select top-k experts
#         topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
#         ### norm gate to sum 1
#         if self.top_k > 1 and self.norm_topk_prob:
#             denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
#             topk_weight = topk_weight / denominator

#         ### expert-level computation auxiliary loss
#         if self.training and self.alpha > 0.0:
#             scores_for_aux = scores
#             aux_topk = self.top_k
#             # always compute aux loss based on the naive greedy topk method
#             topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
#             if self.seq_aux:
#                 scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
#                 ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
#                 ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
#                 aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
#             else:
#                 mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
#                 ce = mask_ce.float().mean(0)
#                 Pi = scores_for_aux.mean(0)
#                 fi = ce * self.n_routed_experts
#                 aux_loss = (Pi * fi).sum() * self.alpha
#         else:
#             aux_loss = None
#         return topk_idx, topk_weight, aux_loss
    
# class PhiMoESparseMoeBlock(nn.Module):
#     """
#     This implementation is
#     strictly equivalent to standard MoE with full capacity (no
#     dropped tokens). It's faster since it formulates MoE operations
#     in terms of block-sparse operations to accomodate imbalanced
#     assignments of tokens to experts, whereas standard MoE either
#     (1) drop tokens at the cost of reduced performance or (2) set
#     capacity factor to number of experts and thus waste computation
#     and memory on padding.
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.hidden_dim = config.hidden_size
#         self.ffn_dim = config.intermediate_size
#         self.num_experts = config.num_local_experts
#         self.top_k = config.num_experts_per_tok
#         global iterations
#         iterations +=1
#         self.iter = iterations
#         # gating
#         self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

#         self.experts = nn.ModuleList([PhiMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

#         # Jitter parameters
#         self.router_jitter_noise = config.router_jitter_noise
#         self.input_jitter_noise = config.input_jitter_noise
        
#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         """ """
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         if self.training and self.input_jitter_noise > 0:
#             hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         # router_logits: (batch * sequence_length, n_experts)
#         # print ( 'moe', self.iter, torch.norm(hidden_states).item())
#         router_logits = self.gate(hidden_states)

#         routing_weights, selected_experts = sparsemixer(
#             router_logits, 
#             top_k=2, 
#             jitter_eps=self.router_jitter_noise, 
#             training=self.training,
#         )

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         # One hot encode the selected experts to create an expert mask
#         # this will be used to easily index which expert is going to be sollicitated
#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

#         # Loop over all available experts in the model and perform the computation on each expert
#         for expert_idx in range(self.num_experts):
#             expert_layer = self.experts[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx])

#             if top_x.shape[0] == 0:
#                 continue

#             # in torch it is faster to index using lists than torch tensors
#             top_x_list = top_x.tolist()
#             idx_list = idx.tolist()

#             # Index the correct hidden states and compute the expert hidden state for
#             # the current expert. We need to make sure to multiply the output hidden
#             # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
#             current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

#             # However `index_add_` only support torch tensors for indexing so we'll use
#             # the `top_x` tensor here.
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         # print ( 'moe', self.iter, torch.norm(final_hidden_states).item())
#         return final_hidden_states, router_logits
    
        
# def find_layers(module, layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock, MoEGate, PhiMoESparseMoeBlock], name='', process_moe_block=False):
#     res = {}
#     # 1. 特别处理 MixtralSparseMoeBlock 模块
#     if isinstance(module, MixtralSparseMoeBlock) or type(module).__name__ == 'MoEGate' or type(module).__name__ == 'PhiMoESparseMoeBlock':
#         # pdb.set_trace()
#         if process_moe_block:
#             # 如果要处理 MixtralSparseMoeBlock，将其自身加入结果
#             res[name] = module
#             # 并且递归处理其子模块
#             for name1, child in module.named_children():
#                 res.update(find_layers(
#                     child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
#                 ))
#         else:
#             # 如果不处理 MixtralSparseMoeBlock，只递归处理其子模块
#             for name1, child in module.named_children():
#                 res.update(find_layers(
#                     child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=False
#                 ))
#         return res  # 处理完 MixtralSparseMoeBlock 后直接返回，避免重复处理
#     # 2. 判断当前模块是否属于其他指定的层类型 (排除 MixtralSparseMoeBlock)
#     elif type(module) in layers or 'gate' in name:
#         res[name] = module
#     # 3. 递归处理其他非指定类型的模块
#     else:
#         for name1, child in module.named_children():
#             res.update(find_layers(
#                 child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
#             ))

#     return res



# class MixtralCompressedExpert(nn.Module):
#     def __init__(self, config: MixtralConfig, low_rank, shared_w1_v, shared_w2_v, shared_w3_v):
#         super().__init__()
#         # 更改: 使用 low_rank 来定义压缩后的维度
#         '''self.w1_u = nn.Linear(config.hidden_size, low_rank, bias=False)
#         self.w2_u = nn.Linear(config.intermediate_size, low_rank, bias=False)
#         self.w3_u = nn.Linear(config.hidden_size, low_rank, bias=False)'''
#         self.w1_u = nn.Linear(low_rank, config.intermediate_size, bias=False)
#         self.w2_u = nn.Linear(low_rank, config.hidden_size, bias=False)
#         self.w3_u = nn.Linear(low_rank, config.intermediate_size, bias=False)
#         nn.init.zeros_(self.w1_u.weight)
#         nn.init.zeros_(self.w1_u.weight)
#         nn.init.zeros_(self.w1_u.weight)

#         # 更改: 使用共享的 svd_v 参数
#         self.shared_w1_v = shared_w1_v
#         self.shared_w2_v = shared_w2_v
#         self.shared_w3_v = shared_w3_v
        
#         self.act_fn = ACT2FN[config.hidden_act]
#         '''
#         w1_u weight shape: torch.Size([14336, 955])
#         w2_u weight shape: torch.Size([4096, 955])
#         w3_u weight shape: torch.Size([14336, 955])
#         shared_w1_v shape: torch.Size([955, 4096])
#         shared_w2_v shape: torch.Size([955, 14336])
#         shared_w3_v shape: torch.Size([955, 4096])
#         '''
#         # print(f"w1_u weight shape: {self.w1_u.weight.shape}")
#         # print(f"w2_u weight shape: {self.w2_u.weight.shape}")
#         # print(f"w3_u weight shape: {self.w3_u.weight.shape}")
#         # print(f"shared_w1_v shape: {self.shared_w1_v.weight.shape}")
#         # print(f"shared_w2_v shape: {self.shared_w2_v.weight.shape}")
#         # print(f"shared_w3_v shape: {self.shared_w3_v.weight.shape}")
       
#     def forward(self, hidden_states):
#         current_hidden_states = self.act_fn(self.w1_u(self.shared_w1_v(hidden_states))) * self.w3_u(self.shared_w3_v(hidden_states))
#         current_hidden_states = self.w2_u(self.shared_w2_v(current_hidden_states))
#         return current_hidden_states
#     # 如果 shared_w1_v 是 (low_rank, hidden_dim)，那么这里需要做矩阵乘法
#     # 先执行 w1_v(hidden_states) 然后再执行 w1_u
#     '''current_hidden_states = self.act_fn(F.linear(hidden_states, self.shared_w1_v)) * F.linear(hidden_states, self.shared_w3_v)
#     current_hidden_states = F.linear(current_hidden_states, self.shared_w2_v.t())
#     return current_hidden_states'''
#     # self.hidden_dim = config.hidden_size
#     # self.ffn_dim = config.intermediate_size
    
    

# class SVD_MixtralSparseMoeBlock(nn.Module):
#     def __init__(self, config, ratio=1):
#         super().__init__()
#         self.hidden_dim = config.hidden_size
#         self.ffn_dim = config.intermediate_size
#         self.num_experts = config.num_local_experts
#         self.top_k = config.num_experts_per_tok
#         self.router_jitter_noise = config.router_jitter_noise
#         self.ratio = ratio
        
#         self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
#         # 更改: 使用 low_rank 来定义压缩后的维度
#         self.low_rank = int(self.ffn_dim * self.hidden_dim * self.ratio / (self.ffn_dim + self.hidden_dim))
        
#         # 更改: 创建共享的 svd_v 参数

#         # todo: 全 0 设置
#         self.shared_w1_v = nn.Linear(self.hidden_dim, self.low_rank, bias=False)
#         self.shared_w2_v = nn.Linear(self.ffn_dim, self.low_rank, bias=False)
#         self.shared_w3_v = nn.Linear(self.hidden_dim, self.low_rank, bias=False)
#         # Initialize weights to zero
#         nn.init.zeros_(self.shared_w1_v.weight)
#         nn.init.zeros_(self.shared_w2_v.weight)
#         nn.init.zeros_(self.shared_w3_v.weight)


#         '''self.shared_w1_v = nn.Parameter(torch.randn(self.low_rank, self.hidden_dim))
#         self.shared_w2_v = nn.Parameter(torch.randn(self.low_rank, self.ffn_dim))
#         self.shared_w3_v = nn.Parameter(torch.randn(self.low_rank, self.hidden_dim))'''

#         # 更改: 在创建专家时传入共享的 svd_v 参数
#         self.experts = nn.ModuleList([
#             MixtralCompressedExpert(config, self.low_rank, self.shared_w1_v, self.shared_w2_v, self.shared_w3_v)
#             for _ in range(self.num_experts)
#         ])
        
#         self.output_router_logits = config.output_router_logits

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)

#         if self.training and self.router_jitter_noise > 0:
#             hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.router_jitter_noise, 1.0 + self.router_jitter_noise)

#         router_logits = self.gate(hidden_states)
        
#         routing_weights = F.softmax(router_logits, dim=-1)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
#         routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

#         for expert_idx in range(self.num_experts):
#             expert_layer = self.experts[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx])
#             if top_x.shape[0] == 0:
#                 continue

#             top_x_list = top_x.tolist()
#             idx_list = idx.tolist()

#             current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return final_hidden_states, router_logits

def print_memory_usage():
    total_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    
    for i in range(total_gpus):
        allocated = torch.cuda.memory_allocated(device=i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device=i) / 1024 / 1024
        total_allocated += allocated
        total_reserved += reserved
        # print(f"GPU {i} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    
    # print(f"Total - Allocated: {total_allocated:.2f} MiB, Reserved: {total_reserved:.2f} MiB")
    
    return total_allocated, total_reserved



def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            input_ids = self.tensors[index]
            return input_ids

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        try:
            test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                # Fallback for newer datasets versions that disallow script-based loading.
                test_data = load_dataset(
                    'ptb_text_only',
                    'penn_treebank',
                    split='test',
                    revision='refs/convert/parquet'
                )
            else:
                raise
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader



def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    import random
    random.seed(seed)
    cache_file = (
        f"cache/{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    nsamples += 1 #############################
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        try:
            traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                traindata = load_dataset(
                    'ptb_text_only',
                    'penn_treebank',
                    split='train',
                    cache_dir=dataset_cache_dir,
                    revision='refs/convert/parquet'
                )
            else:
                raise
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["text"])
    elif name == "dolly":
        traindata = load_dataset("databricks/databricks-dolly-15k", split="train")
        tot_text = "\n\n".join([f"{item['instruction']}\n{item['context']}\n{item['response']}" for item in traindata])
    else:
        raise NotImplementedError
    
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    # sharing V,add
    '''traindataset = []
    total_samples = int(nsamples / sample_ratio)
    for s in range(total_samples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if random.random() < sample_ratio:
            if s % batch_size == 0:
                if s != 0:
                    attention_mask = torch.ones_like(inp)
                    traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
                inp = trainenc.input_ids[:, :seqlen]
            else:
                inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)'''
    
    torch.save(traindataset, cache_file)
    return traindataset




@torch.no_grad()
def ppl_eval_sharing_ori(model, tokenizer, dev, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=16, params_only=False):
    import random
    import numpy as np
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # 获取模型的主设备
    main_device = next(model.parameters()).device
    print(f"Main device: {main_device}")
    # if 'cuda' not in str(main_device):
    #     main_device = dev
    #     model = model.to(main_device)
    if not params_only:
        for dataset in datasets:
            '''if dataset == 'wikitext2':
                # 使用与新代码相同的数据加载方式
                data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
                data = data.input_ids.to(main_device)'''
            # 对于其他数据集，使用原有的加载方式
            data = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
            # data = next(iter(data)).to(main_device)  # 假设 get_test_data 返回一个 DataLoader

            seqlen = model_seq_len
            n_samples = len(data)
            nlls = []

            with tqdm(range(n_samples), desc=f"Evaluating {dataset} - Perplexity") as progress_bar:
                for i in progress_bar:
                    batch = next(iter(data)).to(main_device)

                    allocated, reserved = print_memory_usage()
                    total_allocated_list.append(allocated)
                    total_reserved_list.append(reserved)

                    with torch.no_grad():
                        output = model(batch)
                        logits = output.logits if hasattr(output, "logits") else output[0]

                    # 确保 logits 在正确的设备上
                    logits = logits.to(main_device)
                    shift_logits = logits[:, :-1, :].contiguous().float()
                    shift_labels = batch[:, 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    neg_log_likelihood = loss.float() * seqlen
                    nlls.append(neg_log_likelihood)

                    curr_ppl = _perplexity(nlls, i + 1, seqlen)
                    progress_bar.set_description(f"Evaluating {dataset} - Perplexity {curr_ppl:.3f}")

            ppl = _perplexity(nlls, n_samples, seqlen)
            ppls[dataset] = ppl.item()

    # 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # 检查 SVD 压缩的 Mixtral 特性
    # svd_layers = sum(1 for m in model.modules() if isinstance(m, SVD_MixtralSparseMoeBlock))
    result_str = f"Experiment: {experiment_name}\n"
    if not params_only:
        avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
        avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
        result_str += f"PPL after evaluation: {ppls}\n"
        result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
        result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    
    result_str += f"Total number of parameters: {total_params / 1e9:.2f}B\n"
    result_str += f"Number of trainable parameters: {trainable_params / 1e9:.2f}B\n"
    result_str += f"Number of non-trainable parameters: {non_trainable_params / 1e9:.2f}B\n"
    # result_str += f"Number of SVD compressed Mixtral layers: {svd_layers}\n"

    # print(result_str)
    return result_str


@torch.no_grad()
def ppl_eval_sharing(model, tokenizer, dev, experiment_name, datasets=['wikitext2'], model_seq_len=2048, batch_size=4, params_only=False):
    """
    评估模型的困惑度 (Perplexity)。
    
    Args:
        model: 要评估的模型。
        tokenizer: 对应的分词器。
        dev: 评估设备 (例如 "cuda")。
        experiment_name (str): 实验名称，用于报告。
        datasets (list): 要评估的数据集列表 (例如 ['wikitext2', 'ptb'])。
        model_seq_len (int): 模型的序列长度。
        batch_size (int): 评估时使用的批处理大小。
        params_only (bool): 如果为True，则只计算参数量，跳过PPL评估。

    Returns:
        str: 包含评估结果的格式化字符串。
    """
    print(f"\n--- 开始 PPL 评估 (实验: {experiment_name}) ---")
    
    model.eval()  # 确保模型处于评估模式
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # 自动获取模型所在的主要设备
    main_device = next(model.parameters()).device
    print(f"模型主要运行在设备: {main_device}")

    if not params_only:
        for dataset in datasets:
            # 1. 加载测试数据
            # get_test_data 返回一个 torch.utils.data.DataLoader 对象
            data_loader = get_test_data(
                dataset, 
                tokenizer, 
                seq_len=model_seq_len, 
                batch_size=batch_size
            )
            
            nlls = []
            total_tokens = 0

            # 2. 正确的迭代循环
            progress_bar = tqdm(data_loader, desc=f"正在评估 {dataset}", leave=False)
            for batch in progress_bar:
                batch = batch.to(main_device)
                
                # 基于Token的精确计算
                shift_labels = batch[:, 1:].contiguous()
                tokens_in_batch = shift_labels.numel()
                total_tokens += tokens_in_batch

                # 监控显存使用
                allocated, reserved = print_memory_usage()
                total_allocated_list.append(allocated)
                total_reserved_list.append(reserved)

                # 3. 模型前向传播和损失计算
                outputs = model(batch)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                neg_log_likelihood = loss.float() * tokens_in_batch
                nlls.append(neg_log_likelihood)

                # 动态更新 PPL 到进度条
                current_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
                progress_bar.set_description(f"正在评估 {dataset} | 当前 PPL: {current_ppl:.4f}")

            # 计算最终的 PPL
            final_ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
            ppls[dataset] = final_ppl.item()
            print(f"数据集 '{dataset}' 评估完成, 最终 PPL: {ppls[dataset]:.4f}")

    # 4. 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 5. 格式化并返回最终结果
    result_str = f"\n--- 评估报告: {experiment_name} ---\n"
    if not params_only:
        result_str += f"困惑度 (PPL): {ppls}\n"
        if total_allocated_list:
            avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
            avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
            result_str += f"评估期间平均已分配显存: {avg_allocated:.2f} MiB\n"
            result_str += f"评估期间平均保留显存: {avg_reserved:.2f} MiB\n"
    
    result_str += f"模型总参数量: {total_params / 1e9:.3f} B\n"
    result_str += f"可训练参数量: {trainable_params / 1e9:.3f} B\n"
    result_str += f"非训练参数量: {non_trainable_params / 1e9:.3f} B\n"
    result_str += "--- 报告结束 ---\n"

    return result_str



# def run_lm_eval(model, tokenizer, batch_size=16, task_names=None, output_csv="results.csv", limit=None):      
#     import os
#     import pandas as pd
#     from lm_eval import tasks, evaluator
#     from lm_eval.models.huggingface import HFLM
        
#     if task_names is None:
#         task_names = ["openbookqa", "arc_easy", "winogrande",
#                       "arc_challenge", "piqa",  "mathqa", "hellaswag"]

#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)

#     # 如果已有 CSV，加载已完成的任务
#     finished_tasks = set()
#     if os.path.exists(output_csv):
#         try:
#             prev_df = pd.read_csv(output_csv)
#             finished_tasks = set(prev_df["task"].tolist())
#             print(f"已完成任务: {finished_tasks}")
#         except Exception as e:
#             print(f"读取 {output_csv} 失败，将重新生成: {e}")

#     device = next(model.parameters()).device
#     lm = HFLM(pretrained=model)

#     all_results = []

#     for task in task_names:
#         if task in finished_tasks:
#             print(f"跳过任务 {task} (已存在结果)")
#             continue

#         print(f"\n===== 开始评估任务: {task} =====")
#         results = evaluator.simple_evaluate(
#             model=lm,
#             tasks=[task],       # 每次只跑一个任务
#             batch_size=batch_size,
#             device=device,
#             write_out=False,
#             log_samples=False,
#             verbosity="ERROR",
#             num_fewshot=0,
#             task_manager=tasks.TaskManager(),
#             limit=limit,        # 可选，限制样本数量，加快调试
#         )

#         # 删除 samples，避免结果太大
#         if "samples" in results:
#             del results["samples"]

#         task_results = results["results"].get(task, {})
#         acc = task_results.get("acc,none", None)
#         acc_norm = task_results.get("acc_norm,none", None)

#         row = {
#             "task": task,
#             "acc(%)": f"{acc*100:.2f}%" if acc is not None else "—",
#             "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "—"
#         }

#         print(f"结果: {row}")
#         all_results.append(row)

#         # 追加写入 CSV
#         df = pd.DataFrame([row])
#         if not os.path.exists(output_csv):
#             df.to_csv(output_csv, index=False, mode="w")
#         else:
#             df.to_csv(output_csv, index=False, mode="a", header=False)

#     return pd.DataFrame(all_results)



def run_lm_eval(model, tokenizer, batch_size=16, task_names=None, output_csv="results.csv", limit=None):
    """
    运行 lm-evaluation-harness 评测。
    - 自动从相对路径加载本地的 lm_eval 库。
    - 支持断点续评，结果增量写入 CSV 文件。
    """
    
    import os
    import sys
    import pandas as pd
    from pathlib import Path    
    
    # 动态寻找并切换 lm_eval 库版本
    # print("--- 准备动态切换 lm_eval 库版本 ---")
    # try:
    #     current_file_path = Path(__file__).resolve()
    #     project_root = current_file_path.parent.parent
    #     local_lm_eval_path_obj = project_root / "lm-evaluation-harness"
    #     local_lm_eval_path = str(local_lm_eval_path_obj)

    #     if not local_lm_eval_path_obj.is_dir():
    #         raise FileNotFoundError(f"自动计算的路径不存在: {local_lm_eval_path}")

    #     if local_lm_eval_path not in sys.path:
    #         sys.path.insert(0, local_lm_eval_path)
    #         print(f"已将本地库路径添加到 sys.path: {local_lm_eval_path}")

    # except (NameError, FileNotFoundError) as e:
    #     print(f"无法自动寻找本地 lm_eval 库: {e}")
    #     print("将使用环境中默认安装的版本。")

    # ########### DEBUG ###########
    # import lm_eval 
    # loaded_path = lm_eval.__file__
    # print(f"\n[验证] `lm_eval` 模块实际加载自: {loaded_path}")

    # # 3. 给出明确的判断
    # if local_lm_eval_path in loaded_path:
    #     print("验证成功！当前使用的是你的本地版本。")
    # else:
    #     print("验证失败！当前加载的是系统环境中的其他版本！")
    # print("-" * 20)
    # ########### DEBUG ###########

    from lm_eval import tasks, evaluator
    print("--- lm_eval 库加载完成 ---\n")        

    if task_names is None:
        task_names = ["openbookqa", "arc_easy", "winogrande","arc_challenge", "piqa",  "mathqa", "hellaswag"]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 如果已有 CSV，加载已完成的任务
    finished_tasks = set()
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            finished_tasks = set(prev_df["task"].tolist())
            print(f"已完成任务: {finished_tasks}")
        except Exception as e:
            print(f"读取 {output_csv} 失败，将重新生成: {e}")

    device = next(model.parameters()).device

    all_results = []

    for task in task_names:
        if task in finished_tasks:
            print(f"跳过任务 {task} (已存在结果)")
            continue

        print(f"\n===== 开始评估任务: {task} =====")
        results = evaluator.simple_evaluate(
            model=model,
            tokenizer=tokenizer,
            tasks=[task],       # 每次只跑一个任务
            batch_size=batch_size,
            device=device,
            write_out=False,
            log_samples=False,
            verbosity="ERROR",
            num_fewshot=0,
            task_manager=tasks.TaskManager(),
            limit=limit,        # 可选，限制样本数量，加快调试
        )

        # 删除 samples，避免结果太大
        if "samples" in results:
            del results["samples"]

        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none", None)
        acc_norm = task_results.get("acc_norm,none", None)

        row = {
            "task": task,
            "acc(%)": f"{acc*100:.2f}%" if acc is not None else "—",
            "acc_norm(%)": f"{acc_norm*100:.2f}%" if acc_norm is not None else "—"
        }

        print(f"结果: {row}")
        all_results.append(row)

        # 追加写入 CSV
        df = pd.DataFrame([row])
        if not os.path.exists(output_csv):
            df.to_csv(output_csv, index=False, mode="w")
        else:
            df.to_csv(output_csv, index=False, mode="a", header=False)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python utils.py
    import time
    import gc
    
    model_path="./models/Mixtral-8x7B-Instruct-v0.1"
    model, tokenizer = load_fp16_model(model_path)
    
    
    # --- 3. 运行 PPL 评测 ---
    # dev: 指定评测时主要使用的设备
    model.eval() # 设置为评估模式
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 指定要评测的数据集，可以是 ['wikitext2', 'ptb', 'c4'] 中的一个或多个
    datasets_to_eval = ['ptb']
    
    print(f"Starting PPL evaluation on {datasets_to_eval}...")
    
    # 调用评测函数
    experiment_name="origin"
    results = ppl_eval_sharing_ori(
        model=model,
        tokenizer=tokenizer,
        dev=dev,
        experiment_name=experiment_name,
        datasets=datasets_to_eval,
        batch_size=4 # 根据你的显存调整 batch_size
    )
    print(results)
    
    
    # runt task
    # tasks = ["openbookqa", "arc_easy", "winogrande", "hellaswag","arc_challenge", "piqa", "mathqa"]
    # results = run_lm_eval(model, tokenizer, batch_size=32, task_names=tasks, output_dir="")
    # print(f"{results}")
    # results.to_csv("./output/evaluation_results/acc_origin.csv")
    
    # experiment_name="acc_origin_n_200"
    # print(f"\n--- 开始准确率评估: {experiment_name} ---")
    # results = run_lm_eval(model, tokenizer, batch_size=8, output_dir="") # 限制每个任务的样本200
    # print(f"{results}")
    # results.to_csv(f"./output/evaluation_results/acc_origin_n_200.csv")        
    
    
    # # --- 4. 打印并保存结果 ---
    # print("\n--- Evaluation Finished ---")


    # 可以选择将结果写入文件
    # with open("evaluation_results.txt", "w") as f:
    #     f.write(results)

    # 清理显存
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()            
