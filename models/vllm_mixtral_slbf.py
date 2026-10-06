"""vLLM-compatible Mixtral with SLBF (LRBMoBE) compression on gate+up,
stored in gauge-fixed factored form.

For each MoE layer, gate+up are factored as:
    B_j = U_hat_j V_hat_j^T          (m bases of shape (r, d))
    B_eff[i] = silu(Σ_j W[i,j] · B_j) (per-expert mixed bases)
    gate_proj.T[i] ≈ A[i] · B_eff[i]  ((r, r) @ (r, d))

Mixtral convention (different from Moonlight):
    r = hidden_size       = 4096
    d = intermediate_size = 14336
    Target during SLBF training was gate_proj.T (and up_proj.T), so the
    per-token forward is:
        gate(x)_t = x @ A_gate[i] @ silu(Σ_j W_g_soft[i,j] · B_gate_j)
        out_t     = x @ A_gate[i] @ silu(Σ B_g) → shape (T, intermediate)
    The first silu is the basis-combination (reconstructs gate_proj.T); the
    second silu is Mixtral's SwiGLU activation applied to gate(x) * up(x).

Gauge-fixed storage:
    For each basis j, we pick k pivot rows of U via RRQR (offline, in the
    build script). With the resulting `pivots`, we store
        U_free  (m, r-k, k)   — the non-pivot rows of U_hat
        V_hat   (m, d, k)     — V transformed by the gauge
        pivots  (m, k)        — int32 row indices
    At inference, U_hat is materialized in a transient (m, r, k) buffer by
    scattering I_k at pivot rows and U_free at the rest.

Checkpoint layout (saved by `scripts/build/mixtral/pack_gauge_fixed.py`):
    model.layers.{i}.block_sparse_moe.gate.weight                       (N, hidden)
    model.layers.{i}.block_sparse_moe.U_free_gate                       (m, r-k, k)
    model.layers.{i}.block_sparse_moe.V_hat_gate                        (m, d,   k)
    model.layers.{i}.block_sparse_moe.pivots_gate                       (m, k)  int32
    model.layers.{i}.block_sparse_moe.W_gate                            (N, m)  pre-softmax
    model.layers.{i}.block_sparse_moe.A_gate                            (N, r, r)
    (mirror for up)
    model.layers.{i}.block_sparse_moe.w2                                (N, hidden, intermediate)

Usage:
    from vllm import ModelRegistry
    from models.vllm_mixtral_slbf import MixtralSLBFForCausalLM
    ModelRegistry.register_model("MixtralSLBFForCausalLM", MixtralSLBFForCausalLM)
"""

import os
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

# ── Optional per-stage profiling (mirrors vllm_deepseekv3_slbf.py) ────────
SLBF_PROFILE = bool(int(os.environ.get("SLBF_PROFILE", "0")))
_TIMING_ACCUM: dict[str, float] = defaultdict(float)
_TIMING_COUNT: dict[str, int]   = defaultdict(int)

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


# ── SLBF MoE Block (gauge-fixed gate+up, uncompressed down) ──────────────

class MixtralSLBFMoEBlock(nn.Module):
    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        self.N = config.num_local_experts                  # 8
        self.top_k = config.num_experts_per_tok            # 2
        self.hidden_size = config.hidden_size              # r = 4096
        self.intermediate_size = config.intermediate_size  # d = 14336
        self.r = self.hidden_size
        self.d = self.intermediate_size
        self.m = config.slbf_num_B
        self.k = config.slbf_rank_per_basis

        # Router: replicated linear, no bias.
        self.gate = nn.Linear(self.hidden_size, self.N, bias=False)

        # Gauge-fixed factors.
        # U_hat will be reconstructed transiently as (m, r, k) per forward.
        self.U_free_gate = nn.Parameter(torch.empty(self.m, self.r - self.k, self.k))
        self.V_hat_gate  = nn.Parameter(torch.empty(self.m, self.d, self.k))
        self.W_gate      = nn.Parameter(torch.empty(self.N, self.m))
        self.A_gate      = nn.Parameter(torch.empty(self.N, self.r, self.r))

        self.U_free_up   = nn.Parameter(torch.empty(self.m, self.r - self.k, self.k))
        self.V_hat_up    = nn.Parameter(torch.empty(self.m, self.d, self.k))
        self.W_up        = nn.Parameter(torch.empty(self.N, self.m))
        self.A_up        = nn.Parameter(torch.empty(self.N, self.r, self.r))

        # Pivot + rest indices (not trainable; registered as buffers).
        # rest is stored in the order leftover from RRQR — U_free[i] sits at rest[i].
        self.register_buffer(
            "pivots_gate", torch.zeros(self.m, self.k, dtype=torch.long),
        )
        self.register_buffer(
            "rest_gate",   torch.zeros(self.m, self.r - self.k, dtype=torch.long),
        )
        self.register_buffer(
            "pivots_up",   torch.zeros(self.m, self.k, dtype=torch.long),
        )
        self.register_buffer(
            "rest_up",     torch.zeros(self.m, self.r - self.k, dtype=torch.long),
        )

        # Uncompressed down: stacked per expert (N, hidden, intermediate).
        self.w2 = nn.Parameter(torch.empty(self.N, self.hidden_size, self.intermediate_size))

    def _basis_unmixed_gauge_fixed(self, U_free: torch.Tensor, V_hat: torch.Tensor,
                                    pivots: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
        """Compute B_j = U_hat_j @ V_hat_j^T as (m, r, d) WITHOUT materializing U_hat.

        Exploits the gauge-fixed structure: U_hat has I_k at pivot rows. So
          B[m, pivots[m], :] = V_hat[m].T           (no matmul, just a scatter)
          B[m, rest[m],   :] = U_free[m] @ V_hat[m].T   (smaller matmul: (r-k)·d·k)

        Saves m·k²·d flops vs the naive `einsum(U_hat, V_hat)` (~20% for Mixtral
        k=832, m=8) and avoids the U_hat (m, r, k) allocation.
        """
        m, _, k = U_free.shape
        d = V_hat.shape[1]
        dev, dt = U_free.device, U_free.dtype
        B = torch.empty(m, self.r, d, dtype=dt, device=dev)

        bidx = torch.arange(m, device=dev).unsqueeze(1)
        # Rest rows: smaller batched matmul (m, r-k, k) × (m, k, d)
        B[bidx, rest, :] = torch.einsum("mrk,mdk->mrd", U_free, V_hat)
        # Pivot rows: at identity rows, B == V_hat^T per basis.
        B[bidx, pivots, :] = V_hat.transpose(-1, -2)
        return B

    def _mix_silu(self, W_rows: torch.Tensor, B_unmixed: torch.Tensor) -> torch.Tensor:
        """silu(softmax(W) @ B_unmixed) over m bases, per active expert.

        W_rows: (n_active, m)  pre-softmax mixing weights for active experts
        B_unmixed: (m, r, d)
        returns: (n_active, r, d)
        """
        w_soft = F.softmax(W_rows, dim=-1)
        mixed = w_soft @ B_unmixed.view(self.m, -1)
        return F.silu(mixed).view(-1, self.r, self.d)

    @staticmethod
    def _profile(name):
        if not SLBF_PROFILE:
            return None
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        return (s, e, name)

    @staticmethod
    def _profile_end(tok):
        if tok is None:
            return
        s, e, name = tok
        e.record()
        torch.cuda.current_stream().synchronize()
        _TIMING_ACCUM[name] += s.elapsed_time(e)
        _TIMING_COUNT[name] += 1

    def _route(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states, self.gate.weight)
        scores = F.softmax(router_logits.float(), dim=-1)
        topk_w, selected = torch.topk(scores, self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return selected, topk_w.to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        tok = self._profile("1_route")
        selected, topk_w = self._route(hidden_states)
        self._profile_end(tok)

        # Active experts only (saves basis-mix work).
        active = torch.unique(selected.flatten())
        active_list = active.tolist()

        tok = self._profile("2_basis_unmixed_gf")
        # Fused: skip materializing U_hat; exploit identity at pivot rows.
        B_gate_unmixed = self._basis_unmixed_gauge_fixed(
            self.U_free_gate, self.V_hat_gate, self.pivots_gate, self.rest_gate)
        B_up_unmixed   = self._basis_unmixed_gauge_fixed(
            self.U_free_up,   self.V_hat_up,   self.pivots_up,   self.rest_up)
        self._profile_end(tok)

        tok = self._profile("4_mix")
        B_gate_eff = self._mix_silu(self.W_gate[active], B_gate_unmixed)   # (n_active, r, d)
        B_up_eff   = self._mix_silu(self.W_up[active],   B_up_unmixed)
        self._profile_end(tok)

        tok = self._profile("5_expert_loop")
        output = torch.zeros_like(hidden_states)
        for active_idx, expert_idx in enumerate(active_list):
            mask = (selected == expert_idx)
            token_mask = mask.any(dim=-1)
            if not token_mask.any():
                continue
            t_idx = token_mask.nonzero(as_tuple=False).squeeze(1)
            x = hidden_states[t_idx]                                       # (t, r)
            w = (topk_w[t_idx] * mask[t_idx]).sum(dim=-1, keepdim=True)    # (t, 1)

            ag = self.A_gate[expert_idx]   # (r, r)
            au = self.A_up[expert_idx]
            bg = B_gate_eff[active_idx]    # (r, d)
            bu = B_up_eff[active_idx]

            # x @ A @ silu(B_eff)  — note: silu already applied inside B_eff
            #   (basis-mix silu reconstructs gate_proj.T)
            # Mixtral's SwiGLU then applies silu(g) * u.
            g = (x @ ag) @ bg              # (t, d) -- this is gate(x)
            u = (x @ au) @ bu              # (t, d) -- this is up(x)
            act = F.silu(g) * u            # (t, d)

            wd = self.w2[expert_idx]       # (r, d) -- HF: out_features=hidden, in_features=intermediate
            expert_out = F.linear(act, wd) # (t, r)
            output.index_add_(0, t_idx, expert_out * w)
        self._profile_end(tok)

        return output.view(orig_shape)


# ── Decoder Layer / Model / CausalLM ──────────────────────────────────────

class MixtralSLBFDecoderLayer(nn.Module):
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
        self.block_sparse_moe = MixtralSLBFMoEBlock(
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


class MixtralSLBFModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MixtralSLBFDecoderLayer(
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


class MixtralSLBFForCausalLM(nn.Module, SupportsPP):
    """Mixtral with SLBF gauge-fixed gate+up routed experts."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.model = MixtralSLBFModel(
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
        """Direct name-matched loading; the build script already produced the
        gauge-fixed and stacked-down tensors under canonical names."""
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            # qkv_proj fusion (q/k/v_proj → qkv_proj).
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
                continue

            if name in buffers_dict:
                # pivots are stored as int32 in checkpoint, cast to long for indexing.
                with torch.no_grad():
                    buffers_dict[name].copy_(loaded_weight.to(buffers_dict[name].dtype))
                loaded.add(name)
        return loaded
