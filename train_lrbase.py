import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from safetensors import safe_open
import os
import sys
from safetensors.torch import save_file
import copy
import argparse
import gc

try:
    import wandb
except ImportError:
    wandb = None

# Schedule-free optimizer (local copy)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'schedule_free'))
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None


class LRBMoBE(nn.Module):
    """MoBE with low-rank factored shared bases.

    B_j = U_params[j] @ V_params[j]^T   where U in R^(m,r,k), V in R^(m,d,k).

    Forward:
        B_eff[j,r,d] = einsum('jrk,jdk->jrd', U, V)       # (m, r, d)
        weighted_B   = einsum('bi,ird->brd',  softmax(w), B_eff)  # (B, r, d)
        hat_Z        = A @ f(weighted_B)

    Init strategy (controlled by --init_mode):
      'per_expert'  -- each expert gets its own basis from per-expert SVD (requires num_B==num_matrices)
      'grouped'     -- grouped SVD same as standard MoBE, then LR-factorize each basis

    When k == truncation the LR factorization is full-rank and 'grouped' replicates MoBE exactly.
    """

    def __init__(self, initial_A, initial_U, initial_V, initial_W, activation='silu', uv_normalize=False, w_softmax=True, w_scale=False, b_normalize=False):
        super().__init__()
        self.A_params = nn.Parameter(initial_A)   # (N, r, r)
        self.U_params = nn.Parameter(initial_U)   # (m, r, k)
        self.V_params = nn.Parameter(initial_V)   # (m, d, k)
        self.w_params = nn.Parameter(initial_W)   # (N, m) logits

        if activation not in ('silu', 'tanh', 'none'):
            raise ValueError("activation must be 'silu', 'tanh', or 'none'")
        self.activation = activation
        self.uv_normalize = uv_normalize
        self.w_softmax = w_softmax
        self.w_scale = w_scale
        self.b_normalize = b_normalize

        if w_scale:
            # Per-expert scalar s_e: w_eff = s_e * softmax(logit_e). Init s_e=1.
            N_init = initial_W.size(0)
            self.s_params = nn.Parameter(torch.ones(N_init, 1, device=initial_W.device))  # (N, 1)

        if b_normalize:
            # Per-basis scalar s_j: B_eff_j = s_j * (U_j @ V_j^T) / ||U_j @ V_j^T||_F
            # Init s_j = ||U_j @ V_j^T||_F so the effective B_eff is unchanged at init.
            with torch.no_grad():
                B_init = torch.einsum('jrk,jdk->jrd', initial_U, initial_V)  # (m, r, d)
                frob = B_init.flatten(1).norm(dim=1)                          # (m,)
            self.s_basis = nn.Parameter(frob.to(initial_U.device))            # (m,)

        if uv_normalize:
            with torch.no_grad():
                orig_U_rms = initial_U.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # (m, 1, 1)
                orig_V_rms = initial_V.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # (m, 1, 1)
            self.register_buffer('orig_U_rms', orig_U_rms)
            self.register_buffer('orig_V_rms', orig_V_rms)

        N = initial_A.size(0)
        rows = initial_A.size(1)   # r = rows_per_matrix (output dim of A)
        trunc = initial_A.size(2)  # truncation (inner dim of A, row dim of U)
        m = initial_U.size(0)
        k = initial_U.size(2)
        d = initial_V.size(1)
        n_A = initial_A.numel()
        n_U = initial_U.numel()
        n_V = initial_V.numel()
        n_w = initial_W.numel()
        n_sb = m if b_normalize else 0
        print(f"[lrbase] N={N} m={m} r={rows} trunc={trunc} k={k} d={d}")
        print(f"[lrbase] A={n_A} U={n_U} V={n_V} w={n_w} s_basis={n_sb} total={n_A+n_U+n_V+n_w+n_sb}")
        print(f"[lrbase] B_eff size: {m}x{trunc}x{d} = {m*trunc*d} ({m*trunc*d*4/1e6:.1f} MB)")

    def forward(self, batch_indices):
        A = self.A_params[batch_indices]                                          # (batch, r, trunc)
        if self.w_softmax:
            w = torch.softmax(self.w_params[batch_indices], dim=1)                   # (batch, m)
        else:
            w = self.w_params[batch_indices]                                         # (batch, m)
        if self.w_scale:
            w = w * self.s_params[batch_indices]                                     # (batch, m) scaled by s_e

        if self.uv_normalize:
            U_rms = self.U_params.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            V_rms = self.V_params.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            U_scale = torch.clamp(self.orig_U_rms / U_rms.clamp(min=1e-8), max=1.0)  # min(1, orig/curr)
            V_scale = torch.clamp(self.orig_V_rms / V_rms.clamp(min=1e-8), max=1.0)
            U = self.U_params * U_scale
            V = self.V_params * V_scale
        else:
            U = self.U_params
            V = self.V_params

        B_eff = torch.einsum('jrk,jdk->jrd', U, V)       # (m, r, d)
        if self.b_normalize:
            frob = B_eff.flatten(1).norm(dim=1).clamp(min=1e-8)  # (m,)
            B_eff = B_eff / frob.view(-1, 1, 1) * self.s_basis.view(-1, 1, 1)
        weighted_B = torch.einsum('bi,ird->brd', w, B_eff)                       # (batch, r, d)
        
        if self.activation == 'silu':
            act = F.silu(weighted_B)
        elif self.activation == 'tanh':
            act = torch.tanh(weighted_B)
        else:
            act = weighted_B
        return A @ act                                                             # (batch, r, d)


def _lr_factorize(M, k, device):
    """Rank-k LR factorization of M (r x d): returns U (r,k), V (d,k) with U@V^T ~ M."""
    P, S, Qt = torch.linalg.svd(M.float(), full_matrices=False)  # P:(r,r), S:(r,), Qt:(r,d)
    sqrt_S = S[:k].sqrt()
    U = P[:, :k] * sqrt_S          # (r, k)
    V = Qt[:k, :].T * sqrt_S       # (d, k)
    return U, V


def get_layer_proj_dict(index_path, base_dir, layer_i, matrix_type, model_variant="default"):
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    layer_dict = {}

    if model_variant in ("qwen3_5", "gemma4"):
        # qwen3_5: model.language_model.layers.{i}.mlp.experts.gate_up_proj
        # gemma4 : model.language_model.layers.{i}.experts.gate_up_proj  (no .mlp.)
        if model_variant == "qwen3_5":
            prefix = f"model.language_model.layers.{layer_i}.mlp.experts"
        else:
            prefix = f"model.language_model.layers.{layer_i}.experts"
        if matrix_type in ("gate_proj", "up_proj"):
            # gate_up_proj is fused: shape [num_experts, 2*moe_intermediate_size, hidden_size]
            fused_key = f"{prefix}.gate_up_proj"
            safetensor_file = index_data['weight_map'][fused_key]
            with safe_open(f"{base_dir}/{safetensor_file}", framework="pt") as f:
                fused = f.get_tensor(fused_key)  # [N, 2*rows, cols]
            rows = fused.shape[1] // 2
            split = fused[:, :rows, :] if matrix_type == "gate_proj" else fused[:, rows:, :]
            for i in range(fused.shape[0]):
                layer_dict[f"{prefix}.{i}.{matrix_type}.weight"] = split[i]
        else:  # down_proj: stored as batched [N, hidden, intermediate]
            key = f"{prefix}.{matrix_type}"
            safetensor_file = index_data['weight_map'][key]
            with safe_open(f"{base_dir}/{safetensor_file}", framework="pt") as f:
                tensor = f.get_tensor(key)  # [N, d_orig, r_orig]
            for i in range(tensor.shape[0]):
                layer_dict[f"{prefix}.{i}.{matrix_type}.weight"] = tensor[i]
    elif model_variant == "mixtral":
        # Mixtral HF: model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight
        # w1=gate (out=ffn=14336, in=hidden=4096) → transpose to (4096, 14336)
        # w3=up   same shape → transpose
        # w2=down (out=hidden=4096, in=ffn=14336) → already (4096, 14336), no transpose
        # Transposed-view convention: rows=4096 (= trunc), cols=14336 for all three projs.
        proj_to_w = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        w_name = proj_to_w[matrix_type]
        prefix = f"model.layers.{layer_i}.block_sparse_moe.experts"
        for weight_name, safetensor_file in index_data['weight_map'].items():
            if weight_name.startswith(f"{prefix}.") and weight_name.endswith(f".{w_name}.weight"):
                with safe_open(f"{base_dir}/{safetensor_file}", framework="pt") as f:
                    tensor = f.get_tensor(weight_name)
                if matrix_type in ("gate_proj", "up_proj"):
                    tensor = tensor.T.contiguous()
                # Remap to unified naming so the training loop can find the tensor.
                new_key = weight_name.replace(f".{w_name}.weight", f".{matrix_type}.weight")
                layer_dict[new_key] = tensor
    else:
        for weight_name, safetensor_file in index_data['weight_map'].items():
            if f"model.layers.{layer_i}.mlp.experts" in weight_name and weight_name.endswith(f"{matrix_type}.weight"):
                safetensor_path = f"{base_dir}/{safetensor_file}"
                with safe_open(safetensor_path, framework="pt") as f:
                    layer_dict[weight_name] = f.get_tensor(weight_name)
    return layer_dict


def parse_args():
    parser = argparse.ArgumentParser(description="MoBE Training with Low-Rank Bases")
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--num_hidden_layers", type=int, default=94)
    parser.add_argument("--num_matrices", type=int, default=128)
    parser.add_argument("--rows_per_matrix", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=4096)

    parser.add_argument("--num_epochs", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.07)

    parser.add_argument("--num_B", type=int, default=32, help="Number of bases (m)")
    parser.add_argument("--rank_per_basis", type=int, default=140, help="Rank k per LR basis (U: r x k, V: d x k)")
    parser.add_argument("--k_schedule", type=int, nargs="+", default=None,
                        help="Per-layer k values (one per layer, indexed from 0). Overrides --rank_per_basis. "
                             "Length must cover start_layer..end_layer-1. "
                             "Generate with: python scripts/layer_schedule.py --k K --alpha A")
    parser.add_argument("--truncation", type=int, default=1536, help="Left factor rank (r = rows_per_matrix in practice)")
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=94)

    parser.add_argument("--matrix_type", type=str, choices=["gate_proj", "up_proj", "down_proj"], default="gate_proj")
    parser.add_argument("--model_variant", type=str, choices=["default", "qwen3_5", "gemma4", "mixtral"], default="default",
                        help="'qwen3_5': model.language_model.layers.X.mlp.experts.gate_up_proj (fused). "
                             "'gemma4' : model.language_model.layers.X.experts.gate_up_proj (fused, no .mlp). "
                             "'mixtral': model.layers.X.block_sparse_moe.experts.Y.w{1,2,3}.weight; "
                             "transposed-view (gate/up transposed at load; trunc=hidden_size=4096).")
    parser.add_argument("--activation", type=str, choices=["silu", "tanh", "none"], default="silu")

    parser.add_argument(
        "--init_mode", type=str, choices=["per_expert", "grouped", "msign", "random"], default=None,
        help="'per_expert': one basis per expert (num_B must equal num_matrices); "
             "'grouped': grouped SVD then LR-factorize each basis (standard MoBE style); "
             "'msign': per-expert SVD subspace without spectrum, orthonormal U/V; "
             "'random': A from per-expert SVD, U=[I_k;0], V random orthogonal, w uniform. "
             "Default: per_expert if num_B==num_matrices, else grouped."
    )
    parser.add_argument(
        "--w_init_logit", type=float, default=1.0,
        help="Logit value for own-basis slot at init (others=0). "
             "logit=1.0 -> softmax~0.021 (default, good gradient flow); "
             "logit=7.0 -> softmax~0.90 (near one-hot)."
    )
    parser.add_argument(
        "--init_only", action="store_true",
        help="Compute epoch-0 MSE after init and exit without training."
    )
    parser.add_argument(
        "--ortho_freq", type=int, default=0,
        help="Re-orthogonalize U every N epochs via QR retraction (0 = disabled). "
             "Absorbs R into V to keep B_eff = U@V^T invariant. Fixes gauge ambiguity."
    )
    parser.add_argument(
        "--uv_lr_scale", type=float, default=0.1,
        help="LR scale for U and V params relative to main lr (default 0.1). "
             "Compensates for product structure amplifying effective gradient step."
    )
    parser.add_argument(
        "--w_lr_scale", type=float, default=1.0,
        help="LR scale for A and w params relative to main lr (default 1.0)."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="Weight decay for AdamW (applied to all param groups). 0 = Adam (default)."
    )
    parser.add_argument(
        "--uv_weight_decay", type=float, default=None,
        help="Weight decay applied only to U and V params. Overrides --weight_decay for U/V. "
             "Targets the U@V^T scale ambiguity without penalizing A or w."
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1 (default 0.9).")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2 (default 0.999).")
    parser.add_argument("--uv_normalize", action="store_true", help="Whether to RMS-normalize U and V factors.")
    parser.add_argument("--no_w_softmax", action="store_true", help="Disable softmax on w_params (use raw weights).")
    parser.add_argument("--w_scale", action="store_true", help="Add per-expert scalar s_e: w_eff = s_e * softmax(logit). Init s_e=1.")
    parser.add_argument("--b_normalize", action="store_true", help="Normalize each B_j to unit Frobenius norm, scaled by learnable s_basis_j. Removes U-V scale ambiguity.")
    parser.add_argument("--lr_scheduler", type=str, choices=["none", "cosine", "warmrestart", "schedulefree", "warmup_cosine", "linear"], default="none",
                        help="LR scheduler: cosine=CosineAnnealingLR, warmrestart=CosineAnnealingWarmRestarts, schedulefree=AdamWScheduleFree (no external scheduler), warmup_cosine=linear warmup then CosineAnnealingLR.")
    parser.add_argument("--sf_warmup_steps", type=int, default=0,
                        help="Warmup steps for schedule-free optimizer (default 0).")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Linear warmup steps for warmup_cosine scheduler.")
    parser.add_argument("--warmrestart_t0", type=int, default=5000, help="T_0 for CosineAnnealingWarmRestarts.")
    parser.add_argument("--warmrestart_tmult", type=int, default=2, help="T_mult for CosineAnnealingWarmRestarts.")
    parser.add_argument("--scheduler_eta_min", type=float, default=1e-4, help="eta_min for cosine/warmrestart schedulers.")
    parser.add_argument("--early_stop_patience", type=int, default=2000, help="Early stopping patience. 0 disables early stopping.")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping (0 = disabled).")
    parser.add_argument("--norm_mode", type=str, choices=["global_std", "per_expert_rms", "spectral"], default="global_std",
                        help="Target normalization: global_std (default), per_expert_rms (divide by RMS per expert), spectral (divide by sigma_1 per expert).")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: bf16 autocast (always on) + torch.compile + skip grad/param norm logging.")
    # Keep legacy flag for backward compat
    parser.add_argument("--per_expert_norm", action="store_true", help="Alias for --norm_mode per_expert_rms.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MoBE")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="lrbase")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_log_every", type=int, default=200)
    parser.add_argument("--wandb_save_artifacts", action="store_true")
    return parser.parse_args()


def setup_wandb(args):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError("wandb is not installed.")
    config = vars(args).copy()
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=config,
    )
    run.save(os.path.abspath(__file__), policy="now")
    return run


def main():
    args = parse_args()
    if args.wandb_log_every <= 0:
        raise ValueError("--wandb_log_every must be positive.")

    # Resolve init_mode default
    if args.init_mode is None:
        args.init_mode = 'per_expert' if args.num_B == args.num_matrices else 'grouped'
    if args.init_mode == 'per_expert' and args.num_B != args.num_matrices:
        raise ValueError("--init_mode per_expert requires --num_B == --num_matrices")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"init_mode={args.init_mode}  num_B={args.num_B}  rank_per_basis={args.rank_per_basis}")
    os.makedirs(args.save_path, exist_ok=True)

    wandb_run = setup_wandb(args)

    m = args.num_B
    trunc = args.truncation  # r for left factor

    if args.k_schedule is not None:
        if len(args.k_schedule) < args.end_layer:
            raise ValueError(f"--k_schedule has {len(args.k_schedule)} entries but end_layer={args.end_layer}. "
                             f"Provide at least {args.end_layer} values (one per layer from 0).")
        print(f"Using k_schedule: layers {args.start_layer}-{args.end_layer-1} → k={args.k_schedule[args.start_layer:args.end_layer]}")

    for n in range(args.start_layer, args.end_layer):
        k_lr = args.k_schedule[n] if args.k_schedule is not None else args.rank_per_basis
        best_real_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        print(f'layer: {n}  k={k_lr}')
        layer_step_base = (n - args.start_layer) * (args.num_epochs + 1)

        state_dict = get_layer_proj_dict(args.index_path, args.base_dir, layer_i=n, matrix_type=args.matrix_type, model_variant=args.model_variant)
        if args.model_variant == "qwen3_5":
            expert_key_fmt = f"model.language_model.layers.{n}.mlp.experts.{{i}}.{args.matrix_type}.weight"
        elif args.model_variant == "gemma4":
            expert_key_fmt = f"model.language_model.layers.{n}.experts.{{i}}.{args.matrix_type}.weight"
        elif args.model_variant == "mixtral":
            expert_key_fmt = f"model.layers.{n}.block_sparse_moe.experts.{{i}}.{args.matrix_type}.weight"
        else:
            expert_key_fmt = f"model.layers.{n}.mlp.experts.{{i}}.{args.matrix_type}.weight"
        target_list = []
        for i in range(args.num_matrices):
            target_list.append(state_dict[expert_key_fmt.format(i=i)].to(torch.float16).to(device))
        target = torch.stack(target_list)  # (N, r, d)
        # down_proj is usually stored as (d, r); transpose to (r, d) so it matches gate/up shape.
        # Mixtral 'mixtral' variant already returns gate/up transposed at load and down in (4096, 14336) — all
        # three projections come out aligned, so no further transpose is needed here.
        transpose_target = (args.matrix_type == "down_proj") and args.model_variant != "mixtral"
        if transpose_target:
            target = target.transpose(1, 2).contiguous()  # (N, d_orig, r_orig) -> (N, r, d)
        norm_mode = "per_expert_rms" if args.per_expert_norm else args.norm_mode
        if norm_mode == "per_expert_rms":
            global_target_std = target.float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # (N, 1, 1)
        elif norm_mode == "spectral":
            sigma1 = torch.stack([torch.linalg.matrix_norm(target[e].float(), ord=2)
                                   for e in range(args.num_matrices)])  # (N,)
            global_target_std = sigma1.view(-1, 1, 1)  # (N, 1, 1)
        else:  # global_std
            global_target_std = target.std()  # scalar

        r = args.rows_per_matrix  # = truncation for our experiments
        d = args.cols

        initial_A_list = []
        initial_U_list = []
        initial_V_list = []
        initial_W_list = []

        if args.init_mode == 'per_expert':
            # One basis per expert: each expert's basis initialized from its own SVD.
            # A_e = U_W (r,r) left singular matrix.
            # B_e_init = U_e_basis @ V_e_basis^T = rank-k_lr approx of (A_e^T @ W_e = diag(S_W) @ Vt_W).
            # This gives A_e @ B_e_init = rank-k_lr SVD approx of W_e.
            for e in range(args.num_matrices):
                W_e = target[e].to(torch.float32)  # (r, d)
                U_W, S_W, Vt_W = torch.linalg.svd(W_e, full_matrices=False)
                # U_W: (r, r),  S_W: (r,),  Vt_W: (r, d)

                A_e = U_W[:, :trunc]  # (r, trunc)

                # LR basis: rank-k_lr factorization of the effective right factor diag(S_W)@Vt_W
                sqrt_S = S_W[:k_lr].sqrt()
                U_e_basis = torch.zeros(trunc, k_lr, device=device)
                U_e_basis[:k_lr, :] = torch.diag(sqrt_S)     # (trunc, k_lr); diagonal in top-left block
                V_e_basis = Vt_W[:k_lr, :].T * sqrt_S        # (d, k_lr)
                # Check: A_e @ (U_e_basis @ V_e_basis^T) = rank-k_lr SVD approx of W_e

                # Soft one-hot: logit 1.0 on own basis, others 0.
                # With m=128, softmax([1,0,...,0]) ~ [0.021, 0.008,...] -- allows gradient flow.
                # (logit 10 would saturate softmax to ~[0.994,...], killing w gradients.)
                w_e = torch.zeros(m, device=device)
                w_e[e] = args.w_init_logit

                initial_A_list.append(A_e)
                initial_U_list.append(U_e_basis)
                initial_V_list.append(V_e_basis)
                initial_W_list.append(w_e)

        elif args.init_mode == 'msign':
            # Per-expert SVD subspace without spectrum: A_e=U_W[:,:trunc], U_e=[I_k;0], V_e=V_W[:k]^T.
            # B_e = U_e @ V_e^T — right singular subspace, no singular values.
            # A_e @ B_e = U_W[:,:trunc] @ [I_k;0] @ V_W[:k,:] — correct subspace, optimizer learns scale.
            # With trunc < r, A_e is rectangular (r, trunc) and U_e is (trunc, k_lr).
            # First num_matrices bases: one per expert (own-basis identity init).
            # Extra bases (m > num_matrices): random orthogonal init, zero initial weight.
            for e in range(args.num_matrices):
                W_e = target[e].to(torch.float32)  # (r, d)
                U_W, _, Vt_W = torch.linalg.svd(W_e, full_matrices=False)
                A_e = U_W[:, :trunc]               # (r, trunc)

                U_e = torch.zeros(trunc, k_lr, device=device)
                U_e[:k_lr, :] = torch.eye(k_lr, device=device)   # [I_k; 0], orthonormal columns
                V_e = Vt_W[:k_lr, :].T.contiguous()               # (d, k_lr), orthonormal columns

                w_e = torch.zeros(m, device=device)
                w_e[e] = args.w_init_logit

                initial_A_list.append(A_e)
                initial_U_list.append(U_e)
                initial_V_list.append(V_e)
                initial_W_list.append(w_e)
            # Extra bases beyond num_matrices: msign-style U=[I_k;0], random orthogonal V
            for _ in range(m - args.num_matrices):
                U_e = torch.zeros(trunc, k_lr, device=device)
                U_e[:k_lr, :] = torch.eye(k_lr, device=device)                 # [I_k; 0], same as msign
                V_e = torch.linalg.qr(torch.randn(d, k_lr, device=device))[0]  # (d, k_lr) orthonormal
                initial_U_list.append(U_e)
                initial_V_list.append(V_e)

        elif args.init_mode == 'random':
            # A from per-expert SVD; U=[I_k;0], V=random orthogonal for all m bases; w uniform.
            # Works for any m (< or > num_matrices).
            for e in range(args.num_matrices):
                W_e = target[e].to(torch.float32)
                U_W, _, _ = torch.linalg.svd(W_e, full_matrices=False)
                initial_A_list.append(U_W[:, :trunc])  # (r, trunc)
                initial_W_list.append(torch.full((m,), args.w_init_logit, device=device))
            for _ in range(m):
                U_j = torch.zeros(trunc, k_lr, device=device)
                U_j[:k_lr, :] = torch.eye(k_lr, device=device)                  # [I_k; 0]
                V_j = torch.linalg.qr(torch.randn(d, k_lr, device=device))[0]   # random orthogonal
                initial_U_list.append(U_j)
                initial_V_list.append(V_j)

        elif args.init_mode == 'grouped':
            num_matrices_per_group = args.num_matrices // m
            remainder = args.num_matrices % m
            group_sizes = [num_matrices_per_group + (1 if i < remainder else 0) for i in range(m)]

            start_expert = 0
            for group_i, group_size in enumerate(group_sizes):
                end_expert = start_expert + group_size
                W_group = target[start_expert:end_expert].reshape(-1, d).to(torch.float32)
                start_expert = end_expert

                U_g, S_g, Vt_g = torch.linalg.svd(W_group, full_matrices=False)
                U_k = U_g[:, :trunc]
                S_k = S_g[:trunc]
                Vt_k = Vt_g[:trunc, :]  # (trunc, d) = (r, d)

                group_A = U_k @ torch.diag(S_k)  # (group_size*r, r)

                # LR-factorize the group basis
                U_j, V_j = _lr_factorize(Vt_k, k_lr, device)  # (r, k_lr), (d, k_lr)
                initial_U_list.append(U_j)
                initial_V_list.append(V_j)

                for weight_i in range(group_size):
                    start_r = weight_i * r
                    end_r = (weight_i + 1) * r
                    single_A = group_A[start_r:end_r, :]  # (r, r)
                    w_e = torch.zeros(m, device=device)
                    w_e[group_i] = args.w_init_logit

                    initial_A_list.append(single_A)
                    initial_W_list.append(w_e)

        initial_A = torch.stack(initial_A_list, dim=0)   # (N, r, r)
        initial_U = torch.stack(initial_U_list, dim=0)   # (m, r, k_lr)
        initial_V = torch.stack(initial_V_list, dim=0)   # (m, d, k_lr)
        initial_W = torch.stack(initial_W_list, dim=0)   # (N, m)

        print(f"A={initial_A.shape} U={initial_U.shape} V={initial_V.shape} W={initial_W.shape}")

        model = LRBMoBE(initial_A, initial_U, initial_V, initial_W, activation=args.activation,
                        uv_normalize=args.uv_normalize, w_softmax=not args.no_w_softmax,
                        w_scale=args.w_scale, b_normalize=args.b_normalize).to(device)
        model = torch.compile(model)

        # Verify output shape
        with torch.no_grad():
            test_out = model(torch.arange(2, device=device))
            assert test_out.shape == (2, r, d), f"Shape mismatch: {test_out.shape}"
            print(f"[lrbase] output shape OK: {test_out.shape}")

        # Epoch-0 MSE (full pass, no grad) — chunked to avoid materializing (N, r, d) fp32 all at once
        with torch.no_grad():
            N_exp_preinit = args.num_matrices
            chunk_preinit = min(16, N_exp_preinit)
            sq_sum0, numel0 = 0.0, 0
            for s in range(0, N_exp_preinit, chunk_preinit):
                e = min(s + chunk_preinit, N_exp_preinit)
                idx_c = torch.arange(s, e, device=device)
                out_c = model(idx_c) * global_target_std
                diff = out_c - target[s:e].to(torch.float32)
                sq_sum0 += (diff ** 2).sum().item()
                numel0 += diff.numel()
                del out_c, diff
            init_mse = sq_sum0 / numel0
        print(f"[lrbase] Epoch-0 MSE (with softmax): {init_mse:.6e}  (init_mode={args.init_mode}, m={m}, k={k_lr})")

        # Epoch-0 MSE without softmax — uses raw w_params as weights so one-hot
        # init [1,0,...,0] routes weight exactly 1.0 to own basis (true structure prior).
        # Chunked over experts to avoid materializing (N, r, d) fp32 tensor all at once.
        with torch.no_grad():
            A_all = model.A_params
            w_raw = model.w_params                                          # (N, m) raw logits
            B_eff = torch.einsum('jrk,jdk->jrd', model.U_params, model.V_params)  # (m, r, d)
            N_exp = A_all.size(0)
            chunk = min(16, N_exp)
            sq_sum = 0.0
            numel = 0
            Z_hat_raw_chunks = []
            target_fp32_full = target.to(torch.float32)
            for s in range(0, N_exp, chunk):
                e = min(s + chunk, N_exp)
                weighted_B_raw_c = torch.einsum('bi,ird->brd', w_raw[s:e], B_eff)  # (chunk, r, d)
                if model.activation == 'silu':
                    act_raw_c = F.silu(weighted_B_raw_c)
                elif model.activation == 'tanh':
                    act_raw_c = torch.tanh(weighted_B_raw_c)
                else:
                    act_raw_c = weighted_B_raw_c
                outputs_raw_c = A_all[s:e] @ act_raw_c                  # (chunk, r, d)
                Z_hat_raw_c = outputs_raw_c * global_target_std
                diff = Z_hat_raw_c - target_fp32_full[s:e]
                sq_sum += (diff ** 2).sum().item()
                numel += diff.numel()
                del weighted_B_raw_c, act_raw_c, outputs_raw_c, diff, Z_hat_raw_c
            init_mse_raw = sq_sum / numel
            del target_fp32_full, Z_hat_raw_chunks
        print(f"[lrbase] Epoch-0 MSE (no softmax, raw w): {init_mse_raw:.6e}  (init_mode={args.init_mode}, m={m}, k={k_lr})")

        # Scale-corrected MSE: per-expert optimal α_e = <Â_e, W_e> / ||Â_e||²
        # measures ||α_e·Â_e - W_e||² — removes scale mismatch, isolates subspace error.
        # Chunked over experts; recomputes recon per chunk (no cached Z_hat_raw).
        with torch.no_grad():
            for label in ("w/ softmax", "no softmax"):
                sq_sum_sc = 0.0
                numel_sc = 0
                for s in range(0, N_exp, chunk):
                    e = min(s + chunk, N_exp)
                    W_c = target[s:e].to(torch.float32)
                    if label == "w/ softmax":
                        idx_c = torch.arange(s, e, device=device)
                        recon_c = model(idx_c) * global_target_std
                    else:
                        w_slc = w_raw[s:e]
                        weighted_B_c = torch.einsum('bi,ird->brd', w_slc, B_eff)
                        if model.activation == 'silu':
                            act_c = F.silu(weighted_B_c)
                        elif model.activation == 'tanh':
                            act_c = torch.tanh(weighted_B_c)
                        else:
                            act_c = weighted_B_c
                        recon_c = (A_all[s:e] @ act_c) * global_target_std
                        del weighted_B_c, act_c
                    dot   = (recon_c * W_c).sum(dim=(1, 2))
                    norm2 = (recon_c * recon_c).sum(dim=(1, 2))
                    alpha = dot / norm2.clamp(min=1e-12)
                    residual = alpha[:, None, None] * recon_c - W_c
                    sq_sum_sc += (residual ** 2).sum().item()
                    numel_sc += residual.numel()
                    del recon_c, W_c, residual
                mse_sc = sq_sum_sc / numel_sc
                print(f"[lrbase] Scale-corrected MSE ({label}): {mse_sc:.6e}  (init_mode={args.init_mode}, m={m}, k={k_lr})")
            del B_eff

        if args.init_only:
            print(f"[init_only] Exiting after epoch-0 MSE report.")
            del model, target, target_list, state_dict
            gc.collect()
            torch.cuda.empty_cache()
            if wandb_run is not None:
                wandb_run.summary[f"layer_{n}_init_mse"] = init_mse
            continue

        # U and V have product structure (B_eff = U@V^T), so effective gradient magnitude
        # is proportional to ||V|| and ||U|| respectively. Use a reduced lr to avoid explosion.
        uv_lr = args.learning_rate * args.uv_lr_scale
        w_lr  = args.learning_rate * args.w_lr_scale
        w_group_params = [model.A_params, model.w_params]
        if args.w_scale:
            w_group_params.append(model.s_params)
        if args.b_normalize:
            w_group_params.append(model.s_basis)
        uv_wd = args.uv_weight_decay if args.uv_weight_decay is not None else args.weight_decay
        aw_wd = args.weight_decay
        is_schedulefree = args.lr_scheduler == "schedulefree"
        if is_schedulefree:
            if AdamWScheduleFree is None:
                raise ImportError("schedule_free package not found. Expected at schedule_free/schedulefree/.")
            optimizer = AdamWScheduleFree([
                {'params': w_group_params, 'lr': w_lr, 'weight_decay': aw_wd},
                {'params': [model.U_params, model.V_params], 'lr': uv_lr, 'weight_decay': uv_wd},
            ], betas=(args.beta1, args.beta2), warmup_steps=args.sf_warmup_steps)
            optimizer.train()
            scheduler = None
        else:
            OptCls = torch.optim.AdamW if (args.weight_decay > 0 or uv_wd > 0) else torch.optim.Adam
            optimizer = OptCls([
                {'params': w_group_params, 'lr': w_lr, 'weight_decay': aw_wd},
                {'params': [model.U_params, model.V_params], 'lr': uv_lr, 'weight_decay': uv_wd},
            ], betas=(args.beta1, args.beta2), fused=True)
            if args.lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.scheduler_eta_min)
            elif args.lr_scheduler == "warmrestart":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.warmrestart_t0, T_mult=args.warmrestart_tmult, eta_min=args.scheduler_eta_min)
            elif args.lr_scheduler == "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=args.scheduler_eta_min / args.learning_rate, total_iters=args.num_epochs)
            elif args.lr_scheduler == "warmup_cosine":
                warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps)
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_steps, eta_min=args.scheduler_eta_min)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_steps])
            else:
                scheduler = None
        early_stop_patience = 0 if args.lr_scheduler == "warmrestart" else args.early_stop_patience

        # Tensors for deferred loss accumulation (avoid per-step .item() syncs)
        _loss_buf = torch.zeros(1, device=device)
        _real_loss_buf = torch.zeros(1, device=device)
        _real_mae_buf = torch.zeros(1, device=device)

        for epoch in range(args.num_epochs):
            need_log = (wandb_run is not None and ((epoch + 1) % args.wandb_log_every == 0 or epoch == 0 or (epoch + 1) < 200))
            need_print = ((epoch + 1) % 200 == 0)
            need_stats = need_log or need_print

            optimizer.zero_grad()
            for batch_idx in range(args.num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min((batch_idx + 1) * args.batch_size, args.num_matrices)
                indices = torch.arange(start_idx, end_idx, device=device)
                if len(indices) == 0:
                    continue

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(indices)
                outputs = outputs.float()
                batch_target = target[indices]

                batch_norm = global_target_std[indices] if norm_mode != "global_std" else global_target_std
                Z_scaled = batch_target / batch_norm
                Z_hat_unscaled = outputs * batch_norm

                loss = F.mse_loss(outputs, Z_scaled.to(torch.float32))

                (loss * len(indices) / args.num_matrices).backward()

                if need_stats:
                    real_loss = F.mse_loss(Z_hat_unscaled, batch_target)
                    real_mae = F.l1_loss(Z_hat_unscaled, batch_target)
                    _loss_buf += loss.detach() * len(indices)
                    _real_loss_buf += real_loss.detach() * len(indices)
                    _real_mae_buf += real_mae.detach() * len(indices)

            _clip_val = args.grad_clip if args.grad_clip > 0 else float('inf')
            need_diag = need_stats and not args.fast
            if need_diag:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_val).item()
                # Grad norms
                grad_A_sum = model.A_params.grad.detach().norm().item() if model.A_params.grad is not None else 0.0
                grad_U_sum = model.U_params.grad.detach().norm().item() if model.U_params.grad is not None else 0.0
                grad_V_sum = model.V_params.grad.detach().norm().item() if model.V_params.grad is not None else 0.0
                grad_w_sum = model.w_params.grad.detach().norm().item() if model.w_params.grad is not None else 0.0
            else:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _clip_val)
                total_grad_norm = 0.0
                grad_A_sum = grad_U_sum = grad_V_sum = grad_w_sum = 0.0
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Riemannian retraction: re-orthogonalize U, absorb R into V.
            # Removes GL(k) gauge ambiguity; stabilizes conditioning of U@V^T.
            # Adam state for U and V is reset after retraction to avoid momentum mismatch.
            if args.ortho_freq > 0 and (epoch + 1) % args.ortho_freq == 0:
                with torch.no_grad():
                    Q, R = torch.linalg.qr(model.U_params)          # (m, r, k), (m, k, k)
                    signs = torch.sign(torch.diagonal(R, dim1=1, dim2=2))  # (m, k) sign of R diagonal
                    Q = Q * signs.unsqueeze(1)                        # consistent sign convention
                    R = R * signs.unsqueeze(2)
                    model.U_params.copy_(Q)
                    model.V_params.copy_(torch.bmm(model.V_params, R.mT))  # V @ R^T
                # Reset optimizer momentum for U and V to avoid gradient direction mismatch
                for p in [model.U_params, model.V_params]:
                    if p in optimizer.state:
                        optimizer.state[p]['exp_avg_sq'].zero_()
                        if 'exp_avg' in optimizer.state[p]:  # Adam/AdamW
                            optimizer.state[p]['exp_avg'].zero_()
                        if 'z' in optimizer.state[p]:  # schedule-free: sync z to current p
                            optimizer.state[p]['z'].copy_(p)

            if need_stats:
                epoch_loss = (_loss_buf / args.num_matrices).item()
                epoch_real_loss = (_real_loss_buf / args.num_matrices).item()
                epoch_real_mae = (_real_mae_buf / args.num_matrices).item()
                _loss_buf.zero_()
                _real_loss_buf.zero_()
                _real_mae_buf.zero_()
            else:
                epoch_loss = epoch_real_loss = epoch_real_mae = 0.0

            # Schedule-free: recompute real loss at x-point (averaged iterate).
            # The training pass above measured loss at y (interpolated iterate);
            # eval() switches params to x before we measure and select best model.
            if is_schedulefree:
                optimizer.eval()
                with torch.no_grad():
                    sf_real_loss, sf_real_mae = 0.0, 0.0
                    for _b in range(args.num_batches):
                        _s = _b * args.batch_size
                        _e = min((_b + 1) * args.batch_size, args.num_matrices)
                        _idx = torch.arange(_s, _e, device=device)
                        if len(_idx) == 0:
                            continue
                        _out = model(_idx)
                        _tgt = target[_idx]
                        _norm = global_target_std[_idx] if norm_mode != "global_std" else global_target_std
                        _hat = _out * _norm
                        sf_real_loss += F.mse_loss(_hat, _tgt).item() * len(_idx)
                        sf_real_mae += F.l1_loss(_hat, _tgt).item() * len(_idx)
                optimizer.train()
                epoch_real_loss = sf_real_loss / args.num_matrices
                epoch_real_mae = sf_real_mae / args.num_matrices

            if need_stats and epoch_real_loss < best_real_loss:
                best_real_loss = epoch_real_loss
                best_epoch = epoch
                if is_schedulefree:
                    optimizer.eval()
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                if is_schedulefree:
                    optimizer.train()

            if need_log:
                step = layer_step_base + epoch + 1
                log_dict = {
                    "layer": n,
                    "epoch": epoch + 1,
                    "loss/scaled_mse": epoch_loss,
                    "loss/real_mse": epoch_real_loss,
                    "loss/real_mae": epoch_real_mae,
                    "loss/best_real_mse": best_real_loss,
                    "best_epoch": best_epoch + 1,
                    "optim/lr": optimizer.param_groups[0].get('scheduled_lr', optimizer.param_groups[0]['lr']),
                }
                if need_diag:
                    with torch.no_grad():
                        A_norm = torch.linalg.norm(model.A_params.detach().float(), dim=(1, 2))
                        U_norm = torch.linalg.norm(model.U_params.detach().float(), dim=(1, 2))
                        V_norm = torch.linalg.norm(model.V_params.detach().float(), dim=(1, 2))
                        w_norm = torch.linalg.norm(model.w_params.detach().float(), dim=1)
                        if args.w_scale:
                            s_norm = model.s_params.detach().float().abs().mean().item()
                    log_dict.update({
                        "grad_norm/total": total_grad_norm,
                        "grad_norm/A_mean": grad_A_sum,
                        "grad_norm/U_mean": grad_U_sum,
                        "grad_norm/V_mean": grad_V_sum,
                        "grad_norm/w_mean": grad_w_sum,
                        "A_norm/mean": A_norm.mean().item(),
                        "U_norm/mean": U_norm.mean().item(),
                        "V_norm/mean": V_norm.mean().item(),
                        "w_norm/mean": w_norm.mean().item(),
                        **({"s_scale/mean": s_norm} if args.w_scale else {}),
                        **({"s_basis/mean": model.s_basis.detach().float().mean().item(),
                            "s_basis/std":  model.s_basis.detach().float().std().item()} if args.b_normalize else {}),
                    })
                wandb.log(log_dict, step=step)

            if need_print:
                print(f"Epoch {epoch+1}, Scaled Loss: {epoch_loss:.10f}, Real MSE: {epoch_real_loss:.10f} (best {best_real_loss:.10f} at {best_epoch+1})")
                print(f"Grad [A/U/V/w]: {grad_A_sum:.6f} / {grad_U_sum:.6f} / {grad_V_sum:.6f} / {grad_w_sum:.6f}")

            if early_stop_patience > 0 and (epoch - best_epoch) >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}: best {best_real_loss:.10f} at epoch {best_epoch+1}")
                break

        model.load_state_dict(best_model_state)
        model.eval()
        reconstructed_dict = {}
        with torch.no_grad():
            indices = torch.arange(0, args.num_matrices, device=device)
            outputs = model(indices)
            Z_hat_unscaled = outputs * global_target_std
            for weight_i in range(args.num_matrices):
                key = f'experts_{weight_i}_{args.matrix_type}_weight'
                w = Z_hat_unscaled[weight_i]
                if transpose_target:
                    w = w.transpose(0, 1).contiguous()  # (r, d) -> (d, r) back to original down_proj shape
                reconstructed_dict[key] = w

        # output_path = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_weight.safetensors'
        # save_file(reconstructed_dict, output_path)
        out_file = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_WAB.pth'
        sd = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
        torch.save(sd, out_file)
        print(f"Saved layer {n} at {out_file}, best MSE {best_real_loss:.10f}")

        model_params = sum(p.numel() for p in model.parameters())
        compression_ratio = model_params / (args.num_matrices * args.rows_per_matrix * args.cols)
        print(f'Layer: {n}, Compression Ratio: {compression_ratio:.5f}')

        if wandb_run is not None:
            wandb.log(
                {
                    f"layer/{n}/best_real_mse": best_real_loss,
                    f"layer/{n}/best_epoch": best_epoch + 1,
                    f"layer/{n}/compression_ratio": compression_ratio,
                },
                step=layer_step_base + args.num_epochs,
            )
            wandb_run.summary[f"layer_{n}_best_real_mse"] = best_real_loss

        del target, target_list, state_dict, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
