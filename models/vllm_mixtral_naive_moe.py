"""vLLM-compatible Mixtral with NAIVE (per-expert loop) routed experts.

Baseline for SLBF throughput/VRAM comparison. Uses the FULL uncompressed
weights via a Python for-loop over experts, mirroring the dispatch pattern
of `vllm_mixtral_slbf.py`. The implementation gap vs vLLM's stock FusedMoE
is then isolated separately from the SLBF reconstruction cost.

Differences from stock Mixtral:
    - block_sparse_moe replaced with `MixtralNaiveMoEBlock`
    - per-expert w1/w2/w3 stacked into 3D `nn.Parameter`s (no FusedMoE kernel)
    - Everything else (attention, norms, router gate) is identical.

Checkpoint format: standard Mixtral safetensors.
    model.layers.{i}.block_sparse_moe.gate.weight                   (N, hidden)
    model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight         (intermediate, hidden)
    model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight         (hidden, intermediate)
    model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight         (intermediate, hidden)

Usage:
    from vllm import ModelRegistry
    from models.vllm_mixtral_naive_moe import MixtralNaiveMoEForCausalLM
    ModelRegistry.register_model("MixtralNaiveMoEForCausalLM", MixtralNaiveMoEForCausalLM)
"""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.mixtral import MixtralAttention

logger = init_logger(__name__)


# ── Naive MoE Block (full uncompressed weights, per-expert loop) ──────────

class MixtralNaiveMoEBlock(nn.Module):
    """Per-expert dispatch loop with full (intermediate, hidden) weights."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.N = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size                # 4096
        self.intermediate_size = config.intermediate_size    # 14336

        # Router: replicated linear, no bias.
        self.gate = nn.Linear(self.hidden_size, self.N, bias=False)

        # Stacked per-expert weights (uncompressed, HF orientation).
        self.w1 = nn.Parameter(
            torch.empty(self.N, self.intermediate_size, self.hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.N, self.hidden_size, self.intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(self.N, self.intermediate_size, self.hidden_size)
        )

    def _route(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states, self.gate.weight)            # (T, N)
        scores = F.softmax(router_logits.float(), dim=-1)
        topk_w, selected = torch.topk(scores, self.top_k, dim=-1)            # (T, top_k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return selected, topk_w.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        selected, topk_w = self._route(hidden_states)

        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.N):
            expert_mask = (selected == expert_idx)                            # (T, top_k)
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            tok = token_mask.nonzero(as_tuple=False).squeeze(1)
            x = hidden_states[tok]                                            # (t, hidden)
            w = (topk_w[tok] * expert_mask[tok]).sum(dim=-1, keepdim=True)    # (t, 1)

            g = F.linear(x, self.w1[expert_idx])    # (t, intermediate)
            u = F.linear(x, self.w3[expert_idx])
            act = F.silu(g) * u
            expert_out = F.linear(act, self.w2[expert_idx])   # (t, hidden)
            output.index_add_(0, tok, expert_out * w)

        return output.view(orig_shape)


# ── Decoder Layer / Model / CausalLM ──────────────────────────────────────

class MixtralNaiveMoEDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MixtralAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.block_sparse_moe = MixtralNaiveMoEBlock(
            config, prefix=f"{prefix}.block_sparse_moe",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralNaiveMoEModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MixtralNaiveMoEDecoderLayer(
                config, cache_config, quant_config, prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size,
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            )
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralNaiveMoEForCausalLM(nn.Module, SupportsPP):
    """Naive (per-expert loop) Mixtral baseline."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = MixtralNaiveMoEModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Mixtral checkpoint into stacked 3D per-expert Parameters.

        Per-expert keys (w1/w2/w3) are stacked into `block_sparse_moe.w{1,2,3}[j]`.
        """
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if ".block_sparse_moe.experts." in name:
                # Parse: model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight
                parts = name.split(".")
                try:
                    e_idx = parts.index("experts")
                    expert_j = int(parts[e_idx + 1])
                    proj = parts[e_idx + 2]   # w1, w2, or w3
                except (ValueError, IndexError):
                    continue
                if proj not in ("w1", "w2", "w3") or parts[e_idx + 3] != "weight":
                    continue
                stacked_name = ".".join(parts[:e_idx] + [proj])
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                with torch.no_grad():
                    param.data[expert_j].copy_(loaded_weight)
                loaded.add(stacked_name)
                continue

            # qkv_proj fusion: q/k/v_proj → qkv_proj
            handled = False
            for src, shard_id in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
                if f".{src}." in name:
                    new_name = name.replace(f".{src}.", ".qkv_proj.")
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        loaded.add(new_name)
                        handled = True
                    break
            if handled:
                continue

            if name in params_dict:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        return loaded
