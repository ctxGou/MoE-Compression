"""Build a bf16 HuggingFace proxy model by reconstructing expert weights from WAB.pth.

Handles both MoBE (A, B, w) and LRBMoBE (A, U, V, w) checkpoints.
Reconstruction is done in float32/bf16 precision, then cast to bf16 before saving.

Two modes:
  default (in-memory): loads full base model with .from_pretrained, modifies expert
    weights in-place, then saves. Requires that the model fit in available GPU memory
    (with device_map="auto"). Suitable for ≤~Qwen3-30B / Moonlight on 1-2 A6000s.
  --streaming: walks safetensors shards one at a time, reconstructing only the target
    expert tensors. Loads no full model. Required for very large MoE models (e.g.
    Qwen3.5-122B, DeepSeek-V3) that don't fit on a single node's GPUs. qwen3_5 variant
    only.

Usage:
    # In-memory (default)
    python get_hf_model_from_wab.py \\
        --base_model local_models/Qwen3-30B-A3B-Instruct-2507 \\
        --mobe_dir  results/lrbase_msign_cosine_k147_nb128 \\
        --save_dir  results/lrbase_msign_cosine_k147_nb128_proxy_bf16 \\
        --start_layer 0 --end_layer 48 --num_experts 128

    # Streaming (Qwen3.5-122B)
    python get_hf_model_from_wab.py --streaming \\
        --base_model local_models/Qwen3.5-122B-A10B \\
        --mobe_dir  results/wab/qwen3.5-122b/lrbase_msign_cosine_k250_uniform_nb256_8k \\
        --save_dir  results/proxy_new/qwen3.5-122b/slbf_k250_nb256_proxy \\
        --start_layer 0 --end_layer 48 --num_experts 256 \\
        --model_variant qwen3_5 --gauge_fix
"""

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_FINITE_FULL_SCAN_NUMEL = 1_000_000
_FINITE_SAMPLE_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  required=True)
    parser.add_argument("--mobe_dir",    required=True,
                        help="Default WAB dir; used for any projection lacking a per-proj override.")
    parser.add_argument("--gate_dir",    default=None,
                        help="Per-projection override: WAB dir for gate_proj.")
    parser.add_argument("--up_dir",      default=None,
                        help="Per-projection override: WAB dir for up_proj.")
    parser.add_argument("--down_dir",    default=None,
                        help="Per-projection override: WAB dir for down_proj.")
    parser.add_argument("--skip_projections", nargs="+", default=[],
                        choices=PROJECTIONS,
                        help="Projections to leave UNCOMPRESSED (use original weights). "
                             "Use for 2-of-3 ablations.")
    parser.add_argument("--required_projections", nargs="+", default=None,
                        choices=PROJECTIONS,
                        help="Projections whose WAB artifacts must exist and validate for every "
                             "selected layer before the base model is loaded. Omit to preserve "
                             "the legacy behavior where missing projections pass through unchanged. "
                             "Use all three choices for a strict all-projection proxy build.")
    parser.add_argument("--save_dir",    required=True)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer",   type=int, default=48)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--norm_mode",   default="global_std",
                        choices=["global_std", "per_expert"],
                        help="Must match the norm_mode used during training.")
    parser.add_argument("--activation",  default="silu", choices=["silu", "tanh"])
    parser.add_argument("--gauge_fix",   action="store_true",
                        help="Apply gauge fixing to U/V bases before reconstruction "
                             "(LRBMoBE only; ignored for MoBE with B_params).")
    parser.add_argument("--model_variant", type=str, choices=["default", "qwen3_5", "gemma4", "mixtral"], default="default",
                        help="'qwen3_5': fused gate_up_proj/down_proj under model.language_model.layers.X.mlp.experts. "
                             "'gemma4' : same fused layout but at model.language_model.layers.X.experts (no .mlp). "
                             "'mixtral': model.layers.X.block_sparse_moe.experts.Y.w{1,2,3}.weight; "
                             "transposed-view trained (gate/up rows=4096=hidden, cols=14336=ffn).")
    parser.add_argument("--streaming", action="store_true",
                        help="Walk safetensors shards one at a time without loading full model. "
                             "Required for very large models that don't fit on a single GPU. "
                             "qwen3_5 variant only.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for reconstruction in streaming mode.")
    return parser.parse_args()


def _projection_requires_transpose(proj, model_variant):
    """Whether the HF tensor must be transposed to the WAB training orientation."""
    if model_variant == "mixtral":
        return proj in ("gate_proj", "up_proj")
    return proj == "down_proj"


def _to_internal_projection_orientation(weights, proj, model_variant):
    """Normalize an HF expert tensor to the common WAB ``(N, r, d)`` orientation."""
    if _projection_requires_transpose(proj, model_variant):
        return weights.transpose(-2, -1).contiguous()
    return weights


def _to_hf_projection_orientation(weights, proj, model_variant):
    """Restore a reconstructed WAB tensor to the model's HF parameter orientation."""
    if _projection_requires_transpose(proj, model_variant):
        return weights.transpose(-2, -1).contiguous()
    return weights


def _internal_matrix_shape(hf_shape, proj, model_variant):
    if len(hf_shape) != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(hf_shape)}")
    if _projection_requires_transpose(proj, model_variant):
        return hf_shape[1], hf_shape[0]
    return hf_shape[0], hf_shape[1]


def _projection_dir(args, proj):
    override = {
        "gate_proj": getattr(args, "gate_dir", None),
        "up_proj": getattr(args, "up_dir", None),
        "down_proj": getattr(args, "down_dir", None),
    }[proj]
    return override if override is not None else args.mobe_dir


def _wab_path(args, layer_i, proj):
    return os.path.join(
        _projection_dir(args, proj),
        f"model_layers_{layer_i}_mlp_{proj}_WAB.pth",
    )


def load_original_expert_weights(index_path, base_dir, layer_i, proj, num_experts, dtype,
                                  model_variant="default"):
    """Load expert weights in the canonical WAB ``(N, r, d)`` orientation."""
    with open(index_path) as f:
        index = json.load(f)

    if model_variant in ("qwen3_5", "gemma4"):
        # qwen3_5: model.language_model.layers.{i}.mlp.experts.gate_up_proj
        # gemma4 : model.language_model.layers.{i}.experts.gate_up_proj  (no .mlp)
        prefix = (f"model.language_model.layers.{layer_i}.mlp.experts"
                  if model_variant == "qwen3_5"
                  else f"model.language_model.layers.{layer_i}.experts")
        if proj in ("gate_proj", "up_proj"):
            key = f"{prefix}.gate_up_proj"
            shard = os.path.join(base_dir, index["weight_map"][key])
            with safe_open(shard, framework="pt") as f:
                fused = f.get_tensor(key).to(dtype)   # [N, 2*r, d]
            rows = fused.shape[1] // 2
            weights = fused[:, :rows, :] if proj == "gate_proj" else fused[:, rows:, :]
        else:  # down_proj
            key = f"{prefix}.down_proj"
            shard = os.path.join(base_dir, index["weight_map"][key])
            with safe_open(shard, framework="pt") as f:
                weights = f.get_tensor(key).to(dtype)  # [N, hidden, intermediate]
        return _to_internal_projection_orientation(weights, proj, model_variant)
    elif model_variant == "mixtral":
        # Mixtral HF: model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight
        # w1=gate (out=14336, in=4096) → transpose to (4096, 14336)
        # w3=up   same shape → transpose
        # w2=down (out=4096, in=14336) → already (4096, 14336)
        # Returned shape is (N, 4096, 14336) for all three — matches WAB transposed-view training.
        proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        w_name = proj_to_w[proj]
        tensors = []
        for j in range(num_experts):
            key = f"model.layers.{layer_i}.block_sparse_moe.experts.{j}.{w_name}.weight"
            shard = os.path.join(base_dir, index["weight_map"][key])
            with safe_open(shard, framework="pt") as f:
                tensor = f.get_tensor(key).to(dtype)
            tensors.append(_to_internal_projection_orientation(tensor, proj, model_variant))
        return torch.stack(tensors)   # (N, 4096, 14336)
    else:
        tensors = []
        for j in range(num_experts):
            key = f"model.layers.{layer_i}.mlp.experts.{j}.{proj}.weight"
            shard = os.path.join(base_dir, index["weight_map"][key])
            with safe_open(shard, framework="pt") as f:
                tensors.append(f.get_tensor(key).to(dtype))
        weights = torch.stack(tensors)
        return _to_internal_projection_orientation(weights, proj, model_variant)


def compute_std(weights, norm_mode):
    """Reproduce the global_target_std used during training."""
    if norm_mode == "global_std":
        return weights.float().std()                                      # scalar
    else:  # per_expert
        return weights.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # (N,1,1)


def _lu_pivot_rows(U_all: torch.Tensor):
    """
    Select k pivot rows per basis via LU partial pivoting.
    U_all : (m, r, k)  float64
    Returns pivots_all (m, k), rest_all (m, r-k)  LongTensor on same device.
    """
    m, r, k = U_all.shape
    device = U_all.device

    _, swaps = torch.linalg.lu_factor(U_all)    # swaps: (m, k) 1-based LAPACK
    swaps_cpu = (swaps - 1).cpu().tolist()       # 0-based

    pivots_all = torch.empty(m, k,   dtype=torch.long)
    rest_all   = torch.empty(m, r-k, dtype=torch.long)
    for j in range(m):
        perm = list(range(r))
        for i, p in enumerate(swaps_cpu[j]):
            perm[i], perm[p] = perm[p], perm[i]
        pivots_all[j] = torch.tensor(perm[:k], dtype=torch.long)
        rest_all[j]   = torch.tensor(perm[k:], dtype=torch.long)
    return pivots_all.to(device), rest_all.to(device)


def _rrqr_pivot_rows(U_all: torch.Tensor):
    """
    Select k pivot rows per basis via column-pivoted QR on U^T (rank-revealing).
    Maximises the volume of the selected k×k submatrix — better conditioning
    guarantee than LU partial pivoting.

    U_all : (m, r, k)  float64
    Returns pivots_all (m, k), rest_all (m, r-k)  LongTensor on same device.
    """
    m, r, k = U_all.shape
    device = U_all.device

    # Work on A = U^T  (m, k, r) — we want column pivots of U^T = row pivots of U
    A    = U_all.transpose(-2, -1).clone()       # (m, k, r)
    perm = torch.arange(r, device=device).unsqueeze(0).expand(m, -1).clone()  # (m, r)
    bidx = torch.arange(m, device=device)

    for i in range(k):
        # ── Pivot: column with largest norm in active block ──────────────────
        norms   = A[:, i:, i:].norm(dim=1)       # (m, r-i)
        pivot_j = norms.argmax(dim=1) + i         # (m,)  absolute col index

        # ── Swap columns i ↔ pivot_j (batched, different per basis) ──────────
        tmp                    = A[:, :, i].clone()
        A[:, :, i]             = A[bidx, :, pivot_j]
        A[bidx, :, pivot_j]    = tmp
        tmp_p                  = perm[:, i].clone()
        perm[:, i]             = perm[bidx, pivot_j]
        perm[bidx, pivot_j]    = tmp_p

        # ── Householder reflection to zero A[i+1:, i] ────────────────────────
        v = A[:, i:, i].clone()                  # (m, k-i)
        s = v[:, 0].sign()
        s[s == 0] = 1.0
        v[:, 0] += s * v.norm(dim=1)
        nrm = v.norm(dim=1, keepdim=True).clamp(min=1e-12)
        v  /= nrm
        # A[:, i:, i:] -= 2 v (v^T A[:, i:, i:])
        proj = torch.einsum('mk,mkr->mr', v, A[:, i:, i:])   # (m, r-i)
        A[:, i:, i:] -= 2.0 * v.unsqueeze(2) * proj.unsqueeze(1)

    return perm[:, :k].contiguous(), perm[:, k:].contiguous()


def _gauge_fix_B_eff(U_all: torch.Tensor, V_all: torch.Tensor,
                     store_dtype: torch.dtype,
                     pivot: str = "rrqr") -> torch.Tensor:
    """
    Gauge-fix all m bases and return B_eff (m, r, d) in store_dtype.
    Fully batched on whichever device U_all lives on.

    pivot : "rrqr"  — column-pivoted QR on U^T (maximises submatrix volume)
            "lu"    — LU partial pivoting on U   (cheaper, weaker guarantee)

    For each basis j:
      - Select k pivot rows U1 (k, k), rest U2 (r-k, k)
      - U_free = U2 @ U1^{-1}  stored in store_dtype
      - V_hat  = V[j] @ U1^T   stored in store_dtype
      - B_eff[j] = U_hat @ V_hat^T, U_hat has I_k at pivot rows
    """
    m, r, k = U_all.shape
    device = U_all.device

    U64 = U_all.double()
    if pivot == "rrqr":
        pivots_all, rest_all = _rrqr_pivot_rows(U64)
    else:
        pivots_all, rest_all = _lu_pivot_rows(U64)

    # ── Gather U1 (m, k, k) and U2 (m, r-k, k) ─────────────────────────────
    bidx = torch.arange(m, device=device).unsqueeze(1)  # (m, 1)
    U1 = U64[bidx, pivots_all, :]                        # (m, k,   k)
    U2 = U64[bidx, rest_all,   :]                        # (m, r-k, k)

    # ── Batched solve: U_free = U2 @ U1^{-1} ────────────────────────────────
    U_free = torch.linalg.solve(
        U1.transpose(-2, -1),
        U2.transpose(-2, -1),
    ).transpose(-2, -1).to(store_dtype)                  # (m, r-k, k)

    # ── V_hat = V @ U1^T ─────────────────────────────────────────────────────
    V_hat = torch.bmm(
        V_all.double(), U1.transpose(-2, -1)
    ).to(store_dtype)                                     # (m, d, k)

    # ── Reconstruct U_hat: I_k at pivot rows, U_free elsewhere ──────────────
    U_hat = torch.zeros(m, r, k, dtype=store_dtype, device=device)
    eye_k = torch.eye(k, dtype=store_dtype, device=device).unsqueeze(0).expand(m, -1, -1)
    U_hat[bidx, pivots_all, :] = eye_k
    U_hat[bidx, rest_all,   :] = U_free

    return torch.bmm(U_hat, V_hat.transpose(-2, -1))     # (m, r, d) store_dtype


def reconstruct(wab, activation, norm_mode, orig_weights, device, gauge_fix=False, _timings=None):
    """Reconstruct per-expert weight matrices from WAB parameters.

    Returns bfloat16 tensor of shape (N, r, d) on `device`. Pass device="cuda" for speed.
    _timings: optional dict to accumulate stage times (seconds).
    """
    import time
    def _t(): return time.perf_counter()

    act_fn = F.silu if activation == "silu" else torch.tanh
    bf16 = torch.bfloat16

    t0 = _t()
    A = wab["A_params"].to(device=device, dtype=bf16)      # (N, r, r)
    w = wab["w_params"].to(device=device, dtype=bf16)      # (N, m)
    w_soft = torch.softmax(w, dim=-1)                      # (N, m) bf16
    if _timings is not None: _timings["load_Aw"] = _timings.get("load_Aw", 0) + _t() - t0

    if "B_params" in wab:
        # Standard MoBE
        t0 = _t()
        B = wab["B_params"].to(device=device, dtype=bf16)   # (m, r, d)
        m, r, d = B.shape
        weighted_B = torch.mm(w_soft, B.view(m, -1)).view(-1, r, d)
        if _timings is not None: _timings["weighted_B"] = _timings.get("weighted_B", 0) + _t() - t0
    else:
        # LRBMoBE: B_eff[j] = U[j] @ V[j]^T
        t0 = _t()
        U = wab["U_params"].to(device=device, dtype=torch.float32)   # (m, r, k)
        V = wab["V_params"].to(device=device, dtype=torch.float32)   # (m, d, k)
        if _timings is not None: _timings["load_UV"] = _timings.get("load_UV", 0) + _t() - t0

        t0 = _t()
        if gauge_fix:
            # Run batched gauge fix on same device as reconstruction
            B_eff = _gauge_fix_B_eff(U.to(device), V.to(device), store_dtype=bf16)
        else:
            B_eff = torch.einsum("jrk,jdk->jrd", U.to(bf16), V.to(bf16))  # (m, r, d) bf16
        if _timings is not None: _timings["B_eff"] = _timings.get("B_eff", 0) + _t() - t0

        t0 = _t()
        m, r, d = B_eff.shape
        weighted_B = torch.mm(w_soft, B_eff.view(m, -1)).view(-1, r, d)
        if _timings is not None: _timings["weighted_B"] = _timings.get("weighted_B", 0) + _t() - t0

    t0 = _t()
    outputs = torch.bmm(A, act_fn(weighted_B))               # (N, r, d)
    if _timings is not None: _timings["bmm_A"] = _timings.get("bmm_A", 0) + _t() - t0

    # Scale back to original weight space
    t0 = _t()
    std = compute_std(orig_weights, norm_mode)
    result = (outputs * std.to(device=device, dtype=bf16)).to(bf16)
    if _timings is not None: _timings["scale"] = _timings.get("scale", 0) + _t() - t0

    return result


def _load_wab(wab_path, *, mmap=False):
    """Load a tensor-only WAB checkpoint and strip any torch.compile prefix."""
    load_kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        load_kwargs["mmap"] = True
    try:
        wab = torch.load(wab_path, **load_kwargs)
    except TypeError:
        # PyTorch before weights_only/mmap support. Runtime reconstruction remains
        # compatible, while current PyTorch uses lazy mmap for validation.
        wab = torch.load(wab_path, map_location="cpu")
    except RuntimeError as exc:
        if not mmap or "mmap can only be used" not in str(exc):
            raise
        wab = torch.load(wab_path, map_location="cpu", weights_only=True)
    if not isinstance(wab, dict):
        raise ValueError(f"checkpoint must contain a state dict, got {type(wab).__name__}")
    if any("_orig_mod." in k for k in wab):
        wab = {k.replace("_orig_mod.", ""): v for k, v in wab.items()}
    return wab


def _safetensor_shape(base_dir, weight_map, key):
    if key not in weight_map:
        raise KeyError(f"base-model index has no tensor {key!r}")
    shard_path = os.path.join(base_dir, weight_map[key])
    with safe_open(shard_path, framework="pt") as handle:
        return tuple(handle.get_slice(key).get_shape())


def _expected_internal_projection_shape(weight_map, base_dir, layer_i, proj,
                                        num_experts, model_variant):
    """Read only safetensors metadata and return the expected WAB output shape."""
    if model_variant in ("qwen3_5", "gemma4"):
        prefix = (f"model.language_model.layers.{layer_i}.mlp.experts"
                  if model_variant == "qwen3_5"
                  else f"model.language_model.layers.{layer_i}.experts")
        if proj in ("gate_proj", "up_proj"):
            hf_shape = _safetensor_shape(base_dir, weight_map, f"{prefix}.gate_up_proj")
            if len(hf_shape) != 3 or hf_shape[1] % 2:
                raise ValueError(f"invalid fused gate/up shape {hf_shape}")
            n_experts, rows, in_features = hf_shape
            internal_shape = (n_experts, rows // 2, in_features)
        else:
            hf_shape = _safetensor_shape(base_dir, weight_map, f"{prefix}.down_proj")
            if len(hf_shape) != 3:
                raise ValueError(f"invalid down projection shape {hf_shape}")
            internal_shape = (hf_shape[0], hf_shape[2], hf_shape[1])
    elif model_variant == "mixtral":
        proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        keys = [
            f"model.layers.{layer_i}.block_sparse_moe.experts.{j}."
            f"{proj_to_w[proj]}.weight"
            for j in range(num_experts)
        ]
        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise KeyError(f"base-model index has no tensor {missing[0]!r}")
        matrix_shape = _safetensor_shape(base_dir, weight_map, keys[0])
        internal_shape = (num_experts, *_internal_matrix_shape(
            matrix_shape, proj, model_variant))
    else:
        keys = [
            f"model.layers.{layer_i}.mlp.experts.{j}.{proj}.weight"
            for j in range(num_experts)
        ]
        missing = [key for key in keys if key not in weight_map]
        if missing:
            raise KeyError(f"base-model index has no tensor {missing[0]!r}")
        matrix_shape = _safetensor_shape(base_dir, weight_map, keys[0])
        internal_shape = (num_experts, *_internal_matrix_shape(
            matrix_shape, proj, model_variant))

    if internal_shape[0] != num_experts:
        raise ValueError(
            f"base model has {internal_shape[0]} experts, expected {num_experts}")
    return internal_shape


def _has_only_finite_values(tensor):
    """Check all small tensors and a bounded deterministic sample of large tensors."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return False
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= _FINITE_FULL_SCAN_NUMEL:
        values = flat
    else:
        sample_size = min(_FINITE_SAMPLE_SIZE, flat.numel())
        indices = torch.arange(sample_size, dtype=torch.int64)
        indices = indices * (flat.numel() - 1) // max(sample_size - 1, 1)
        values = flat[indices]
    return bool(torch.isfinite(values).all().item())


def _wab_validation_errors(wab, expected_shape):
    errors = []
    n_experts, rows, cols = expected_shape

    required_common = ("A_params", "w_params")
    for key in required_common:
        if key not in wab:
            errors.append(f"missing tensor {key}")

    has_b = "B_params" in wab
    has_u = "U_params" in wab
    has_v = "V_params" in wab
    if has_b and (has_u or has_v):
        errors.append("contains both B_params and U_params/V_params representations")
    elif not has_b and not (has_u and has_v):
        errors.append("requires B_params or both U_params and V_params")

    tensors = {key: value for key, value in wab.items()
               if key in ("A_params", "w_params", "B_params", "U_params", "V_params")}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{key} is not a tensor")
        elif not tensor.is_floating_point():
            errors.append(f"{key} must be floating point, got {tensor.dtype}")
        elif not _has_only_finite_values(tensor):
            qualifier = ("sampled " if tensor.numel() > _FINITE_FULL_SCAN_NUMEL else "")
            errors.append(f"{key} contains non-finite values in its {qualifier}validation")

    A = wab.get("A_params")
    w = wab.get("w_params")
    if isinstance(A, torch.Tensor) and tuple(A.shape) != (n_experts, rows, rows):
        errors.append(
            f"A_params shape {tuple(A.shape)} != {(n_experts, rows, rows)}")
    if isinstance(w, torch.Tensor):
        if w.ndim != 2 or w.shape[0] != n_experts or w.shape[1] <= 0:
            errors.append(
                f"w_params shape {tuple(w.shape)} must be ({n_experts}, m) with m > 0")

    if has_b and isinstance(w, torch.Tensor):
        B = wab.get("B_params")
        expected = (w.shape[1], rows, cols) if w.ndim == 2 else None
        if isinstance(B, torch.Tensor) and expected is not None and tuple(B.shape) != expected:
            errors.append(f"B_params shape {tuple(B.shape)} != {expected}")
    elif has_u and has_v and isinstance(w, torch.Tensor):
        U = wab.get("U_params")
        V = wab.get("V_params")
        if w.ndim == 2 and isinstance(U, torch.Tensor) and isinstance(V, torch.Tensor):
            rank = U.shape[2] if U.ndim == 3 else None
            expected_u = (w.shape[1], rows, rank) if rank is not None else None
            expected_v = (w.shape[1], cols, rank) if rank is not None else None
            if rank is None or rank <= 0 or rank > min(rows, cols):
                errors.append(f"U_params has invalid shape/rank {tuple(U.shape)}")
            elif tuple(U.shape) != expected_u:
                errors.append(f"U_params shape {tuple(U.shape)} != {expected_u}")
            if expected_v is not None and tuple(V.shape) != expected_v:
                errors.append(f"V_params shape {tuple(V.shape)} != {expected_v}")
    return errors


def validate_required_wab_artifacts(args, index_path=None):
    """Validate explicitly required layer/projection artifacts before model loading.

    The legacy default is intentionally a no-op. Required checkpoints are opened one
    at a time with mmap; shape checks touch metadata, and finiteness checks read all
    small tensors but only a bounded sample from large factor tensors.
    """
    required = tuple(dict.fromkeys(getattr(args, "required_projections", None) or ()))
    if not required:
        return 0

    errors = []
    invalid_names = [proj for proj in required if proj not in PROJECTIONS]
    if invalid_names:
        errors.append(f"unknown required projections: {', '.join(invalid_names)}")
    overlap = sorted(set(required).intersection(getattr(args, "skip_projections", ()) or ()))
    if overlap:
        errors.append(
            f"projections cannot be both required and skipped: {', '.join(overlap)}")
    if args.start_layer < 0 or args.end_layer <= args.start_layer:
        errors.append(
            f"invalid layer range [{args.start_layer}, {args.end_layer})")

    entries = []
    for layer_i in range(args.start_layer, args.end_layer):
        for proj in required:
            if proj not in PROJECTIONS:
                continue
            path = _wab_path(args, layer_i, proj)
            entries.append((layer_i, proj, path))
            if not os.path.isfile(path):
                errors.append(f"layer {layer_i} {proj}: missing artifact {path}")
            elif os.path.getsize(path) == 0:
                errors.append(f"layer {layer_i} {proj}: empty artifact {path}")

    index_path = index_path or os.path.join(args.base_model, "model.safetensors.index.json")
    try:
        with open(index_path) as handle:
            weight_map = json.load(handle)["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read base-model weight index {index_path}: {exc}")
        weight_map = None

    if weight_map is not None:
        for layer_i, proj, path in entries:
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            try:
                expected_shape = _expected_internal_projection_shape(
                    weight_map, args.base_model, layer_i, proj, args.num_experts,
                    args.model_variant)
                wab = _load_wab(path, mmap=True)
                artifact_errors = _wab_validation_errors(wab, expected_shape)
                errors.extend(
                    f"layer {layer_i} {proj} ({path}): {message}"
                    for message in artifact_errors
                )
                del wab
            except (OSError, KeyError, RuntimeError, ValueError,
                    EOFError, pickle.UnpicklingError) as exc:
                errors.append(f"layer {layer_i} {proj} ({path}): {exc}")

    if errors:
        details = "\n".join(f"  - {message}" for message in errors)
        raise ValueError(
            "Required WAB artifact validation failed before model build:\n" + details)
    return len(entries)


def streaming_qwen3_5(args, src_dir, save_dir, device):
    """Shard-by-shard reconstruction without loading the full model.

    For qwen3_5 fused gate_up_proj/down_proj layout. Each target tensor is processed
    using its in-shard original (for std), the corresponding WAB checkpoint is loaded
    on demand, reconstructed on GPU, and written into the new shard.
    """
    src = Path(src_dir)
    dst = Path(save_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files (config, tokenizer, etc.)
    for item in src.iterdir():
        if item.suffix == '.safetensors':
            continue
        target = dst / item.name
        if target.exists():
            continue
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target)

    with open(src / 'model.safetensors.index.json') as f:
        weight_map = json.load(f)['weight_map']

    layer_range = range(args.start_layer, args.end_layer)
    target_keys = {}
    for i in layer_range:
        target_keys[f'model.language_model.layers.{i}.mlp.experts.gate_up_proj'] = (i, 'gate_up_proj')
        target_keys[f'model.language_model.layers.{i}.mlp.experts.down_proj']    = (i, 'down_proj')

    # Group keys by shard
    shard_keys = {}
    for key, shard in weight_map.items():
        shard_keys.setdefault(shard, []).append(key)

    n_shards = len(shard_keys)
    phase_start = time.time()

    for s_idx, (shard_name, keys_in_shard) in enumerate(sorted(shard_keys.items())):
        src_shard = src / shard_name
        dst_shard = dst / shard_name
        if dst_shard.exists() and dst_shard.stat().st_size > 0:
            print(f'[shard {s_idx+1}/{n_shards}] {shard_name}: skip (already exists)', flush=True)
            continue

        t0 = time.time()
        out_tensors = {}
        n_modified = 0

        with safe_open(str(src_shard), framework='pt', device='cpu') as f:
            for key in keys_in_shard:
                t = f.get_tensor(key)
                if key not in target_keys:
                    out_tensors[key] = t.to(torch.bfloat16) if t.is_floating_point() and t.dtype != torch.bfloat16 else t
                    continue

                layer_i, proj_type = target_keys[key]

                if proj_type == 'gate_up_proj':
                    rows = t.shape[1] // 2
                    gate_orig = t[:, :rows, :].contiguous()
                    up_orig   = t[:, rows:, :].contiguous()
                    pieces = []
                    for orig_part, proj in [(gate_orig, 'gate_proj'),
                                             (up_orig, 'up_proj')]:
                        wab_path = _wab_path(args, layer_i, proj)
                        name = proj.removesuffix('_proj')
                        if proj in args.skip_projections:
                            pieces.append(orig_part.to(torch.bfloat16))
                            continue
                        if not os.path.exists(wab_path):
                            print(f'  WARN: missing WAB for layer {layer_i} {name}, passing through original', flush=True)
                            pieces.append(orig_part.to(torch.bfloat16))
                            continue
                        wab = _load_wab(wab_path)
                        recon = reconstruct(wab, args.activation, args.norm_mode,
                                            orig_part.float(), device,
                                            gauge_fix=args.gauge_fix)
                        pieces.append(recon.cpu())
                        del wab, recon
                    out_tensors[key] = torch.cat(pieces, dim=1)
                    n_modified += 1
                    print(f'  layer {layer_i:>2} gate_up_proj reconstructed', flush=True)

                else:  # down_proj
                    # Stored shape (N, hidden, intermediate); training treats as (N, intermediate, hidden)
                    down_path = _wab_path(args, layer_i, 'down_proj')
                    if 'down_proj' in args.skip_projections or not os.path.exists(down_path):
                        # No down compression — pass through
                        out_tensors[key] = t.to(torch.bfloat16) if t.is_floating_point() and t.dtype != torch.bfloat16 else t
                        continue
                    down_orig_internal = _to_internal_projection_orientation(
                        t, 'down_proj', 'qwen3_5')
                    wab = _load_wab(down_path)
                    recon = reconstruct(wab, args.activation, args.norm_mode,
                                        down_orig_internal.float(), device,
                                        gauge_fix=args.gauge_fix)
                    out_tensors[key] = _to_hf_projection_orientation(
                        recon.cpu(), 'down_proj', 'qwen3_5')
                    n_modified += 1
                    del wab, recon
                    print(f'  layer {layer_i:>2} down_proj reconstructed', flush=True)

                torch.cuda.empty_cache()

        save_file(out_tensors, str(dst_shard), metadata={'format': 'pt'})
        del out_tensors
        elapsed = time.time() - t0
        total_elapsed = time.time() - phase_start
        eta = (total_elapsed / (s_idx + 1)) * (n_shards - (s_idx + 1))
        print(f'[shard {s_idx+1}/{n_shards}] {shard_name}: {n_modified} modified  '
              f'{elapsed:.1f}s  total {total_elapsed/60:.1f}min  eta {eta/60:.1f}min',
              flush=True)

    print(f'\nDone. Output: {dst}', flush=True)


def main():
    args = parse_args()
    index_path = os.path.join(args.base_model, "model.safetensors.index.json")

    validated = validate_required_wab_artifacts(args, index_path)
    if validated:
        print(f"Validated {validated} required WAB artifacts.", flush=True)

    if args.streaming:
        if args.model_variant != "qwen3_5":
            raise NotImplementedError(
                "Streaming mode is currently only implemented for qwen3_5 variant. "
                "For default (per-expert) variant, the model is small enough to load "
                "in-memory; remove --streaming.")
        device = torch.device(args.device)
        print(f'=== streaming mode: {args.base_model} -> {args.save_dir} ===', flush=True)
        print(f'  variant=qwen3_5  mobe_dir={args.mobe_dir}  '
              f'gauge_fix={args.gauge_fix}  layers=[{args.start_layer}..{args.end_layer-1}]',
              flush=True)
        streaming_qwen3_5(args, args.base_model, args.save_dir, device)
        return

    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    # Build a map of key -> live parameter tensor (not a copy)
    param_map = {k: v for k, v in model.named_parameters()}

    import time
    _timings = {}   # accumulated across all projections of layer 0 for a quick report

    for i in tqdm(range(args.start_layer, args.end_layer), desc="Reconstructing layers"):
        _layer_t0 = time.perf_counter()
        # Use the device of this layer's parameters for reconstruction (matches device_map split)
        if args.model_variant == "qwen3_5":
            _layer_dev_key = f"model.language_model.layers.{i}.mlp.experts.gate_up_proj"
        elif args.model_variant == "gemma4":
            _layer_dev_key = f"model.language_model.layers.{i}.experts.gate_up_proj"
        elif args.model_variant == "mixtral":
            # transformers >=5 loads Mixtral with a fused gate_up_proj/down_proj layout
            # at model.layers.{i}.mlp.experts.* (legacy block_sparse_moe.experts.{j}.w{1,2,3}
            # remains only in on-disk safetensors).
            _layer_dev_key = f"model.layers.{i}.mlp.experts.gate_up_proj"
        else:
            _layer_dev_key = f"model.layers.{i}.mlp.experts.0.gate_proj.weight"
        recon_device = param_map[_layer_dev_key].device if _layer_dev_key in param_map \
                       else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        for proj in PROJECTIONS:
            if proj in args.skip_projections:
                continue  # explicit ablation: leave this projection uncompressed
            wab_path = _wab_path(args, i, proj)
            if not os.path.exists(wab_path):
                continue  # no WAB for this proj (e.g. down_proj not compressed)

            t0 = time.perf_counter()
            wab = _load_wab(wab_path)
            _timings["load_wab"] = _timings.get("load_wab", 0) + time.perf_counter() - t0

            # Original weights for std computation (kept on CPU)
            t0 = time.perf_counter()
            orig = load_original_expert_weights(
                index_path, args.base_model, i, proj, args.num_experts, torch.float32,
                model_variant=args.model_variant,
            )  # (N, r, d)
            _timings["load_orig"] = _timings.get("load_orig", 0) + time.perf_counter() - t0

            # Reconstruct on the same GPU as this layer's parameters (avoids cross-device sync)
            t0 = time.perf_counter()
            reconstructed = reconstruct(wab, args.activation, args.norm_mode, orig, recon_device,
                                        gauge_fix=args.gauge_fix, _timings=_timings)
            reconstructed = reconstructed.cpu()
            _timings["reconstruct_total"] = _timings.get("reconstruct_total", 0) + time.perf_counter() - t0
            del wab, orig

            # Copy directly into live parameter buffers (no new GPU allocation)
            t0 = time.perf_counter()
            if args.model_variant in ("qwen3_5", "gemma4"):
                experts_prefix = (f"model.language_model.layers.{i}.mlp.experts"
                                  if args.model_variant == "qwen3_5"
                                  else f"model.language_model.layers.{i}.experts")
                if proj in ("gate_proj", "up_proj"):
                    fused_key = f"{experts_prefix}.gate_up_proj"
                    rows = reconstructed.shape[1]
                    offset = 0 if proj == "gate_proj" else rows
                    param_map[fused_key].data[:, offset:offset + rows, :].copy_(reconstructed)
                else:  # down_proj: reconstructed is (N, r, d), fused stores (N, d, r)
                    down_key = f"{experts_prefix}.down_proj"
                    param_map[down_key].data.copy_(_to_hf_projection_orientation(
                        reconstructed, proj, args.model_variant))
            elif args.model_variant == "mixtral":
                # transformers >=5: Mixtral loads with fused gate_up_proj (N, 2*ffn, hidden)
                # and batched down_proj (N, hidden, ffn).
                # Reconstructed is (N, hidden=4096, ffn=14336) in transposed-view convention.
                experts_prefix = f"model.layers.{i}.mlp.experts"
                if proj in ("gate_proj", "up_proj"):
                    fused_key = f"{experts_prefix}.gate_up_proj"
                    # Transpose reconstructed (N, hidden, ffn) -> (N, ffn, hidden) to match fused layout.
                    src = _to_hf_projection_orientation(
                        reconstructed, proj, args.model_variant)
                    rows = src.shape[1]   # = ffn = 14336
                    offset = 0 if proj == "gate_proj" else rows
                    param_map[fused_key].data[:, offset:offset + rows, :].copy_(src)
                else:  # down_proj: reconstructed (N, hidden, ffn) matches fused down layout (N, hidden, ffn)
                    down_key = f"{experts_prefix}.down_proj"
                    param_map[down_key].data.copy_(reconstructed)
            else:
                for j in range(args.num_experts):
                    key = f"model.layers.{i}.mlp.experts.{j}.{proj}.weight"
                    w = _to_hf_projection_orientation(
                        reconstructed[j], proj, args.model_variant)
                    param_map[key].data.copy_(w)
            _timings["copy_to_gpu"] = _timings.get("copy_to_gpu", 0) + time.perf_counter() - t0
            del reconstructed

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_elapsed = time.perf_counter() - _layer_t0
        if i == args.start_layer:
            print(f"\n[timing] layer {i} total: {layer_elapsed:.1f}s")
            for stage, t in _timings.items():
                print(f"  {stage:<22s}: {t:.2f}s")
            _timings = {}   # reset after first layer report

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving to {args.save_dir} ...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    # Copy multimodal/processor configs that save_pretrained may not include.
    # Required for vllm to successfully spin up multimodal classes (e.g. Gemma-4).
    import shutil
    for aux in ("processor_config.json", "preprocessor_config.json", "chat_template.jinja"):
        src = os.path.join(args.base_model, aux)
        dst = os.path.join(args.save_dir, aux)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  copied {aux}")
    print("Done.")


if __name__ == "__main__":
    main()
