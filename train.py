import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from safetensors import safe_open
import os
from safetensors.torch import save_file
import copy
import argparse
import gc

try:
    import wandb
except ImportError:
    wandb = None

class MoBE(nn.Module):
    def __init__(self, initial_A, initial_B, initial_W, activation='silu'):
        super().__init__()
        self.A_params = nn.Parameter(initial_A)
        self.B_params = nn.Parameter(initial_B)
        self.w_params = nn.Parameter(initial_W)

        if activation not in ('silu', 'tanh'):
            raise ValueError("activation must be 'silu' or 'tanh'")
        self.activation = activation

    def forward(self, batch_indices):
        A = self.A_params[batch_indices]
        w = self.w_params[batch_indices]
        w = torch.softmax(w, dim=1)
        weighted_B = torch.einsum('bi,ijk->bjk', w, self.B_params)

        if self.activation == 'silu':
            activation = nn.functional.silu(weighted_B)
        else:  
            activation = torch.tanh(weighted_B)

        hat_Z = A @ activation
        return hat_Z


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
        else:  # down_proj is stored as one batched [N, hidden, intermediate] tensor
            down_key = f"{prefix}.down_proj"
            safetensor_file = index_data['weight_map'][down_key]
            with safe_open(f"{base_dir}/{safetensor_file}", framework="pt") as f:
                down = f.get_tensor(down_key)
            for i in range(down.shape[0]):
                layer_dict[f"{prefix}.{i}.down_proj.weight"] = down[i]
    elif model_variant == "mixtral":
        # Mixtral HF: model.layers.{i}.block_sparse_moe.experts.{j}.w{1,2,3}.weight
        # w1 = gate_proj  (out=ffn=14336, in=hidden=4096) → transpose to (4096, 14336)
        # w3 = up_proj    same shape → transpose
        # w2 = down_proj  (out=hidden=4096, in=ffn=14336) → already (4096, 14336), no transpose
        # Transposed view unifies: all three projs trained with rows=4096, cols=14336.
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


def orient_projection_for_training(target, matrix_type, model_variant="default"):
    """Return expert weights in the common [experts, rows, cols] training orientation."""
    if matrix_type == "down_proj" and model_variant != "mixtral":
        return target.transpose(1, 2).contiguous()
    return target


def orient_projection_for_checkpoint(weight, matrix_type, model_variant="default"):
    """Restore one reconstructed weight to its source-checkpoint orientation."""
    if matrix_type == "down_proj" and model_variant != "mixtral":
        return weight.transpose(0, 1).contiguous()
    return weight


def parse_args():
    parser = argparse.ArgumentParser(description="MoBE Training")
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

    parser.add_argument("--num_B", type=int, default=32)
    parser.add_argument("--truncation", type=int, default=1536)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=94)

    parser.add_argument("--matrix_type", type=str, choices=["gate_proj", "up_proj", "down_proj"], default="gate_proj")
    parser.add_argument("--model_variant", type=str, choices=["default", "qwen3_5", "gemma4", "mixtral"], default="default",
                        help="'qwen3_5': model.language_model.layers.X.mlp.experts.gate_up_proj (fused). "
                             "'gemma4' : model.language_model.layers.X.experts.gate_up_proj (fused, no .mlp). "
                             "'mixtral': model.layers.X.block_sparse_moe.experts.Y.w{1,2,3}.weight; "
                             "transposed view (gate/up transposed at load; trunc=4096=hidden_size).")

    parser.add_argument("--activation", type=str, choices=["silu", "tanh"], default="silu")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="MoBE")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default="train2")
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_log_every", type=int, default=200, help="Log W&B metrics every N epochs.")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: bf16 autocast, torch.compile, and skip grad/param norm logging.")
    parser.add_argument(
        "--wandb_save_artifacts",
        action="store_true",
        help="Upload saved layer checkpoints and safetensors as W&B artifacts.",
    )
    return parser.parse_args()


def setup_wandb(args):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --wandb.")

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.save_path, exist_ok=True)

    wandb_run = setup_wandb(args)
    if wandb_run is not None:
        wandb_run.summary["device"] = str(device)

    num_group = args.num_B
    k = args.truncation
    # Distribute remainder across first r groups (each gets one extra expert)
    _base = args.num_matrices // num_group
    _rem  = args.num_matrices %  num_group
    group_sizes   = [_base + (1 if i < _rem else 0) for i in range(num_group)]
    group_offsets = [sum(group_sizes[:i]) for i in range(num_group + 1)]

    for n in range(args.start_layer, args.end_layer):
        best_real_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        print(f'layer: {n}')
        layer_step_base = (n - args.start_layer) * (args.num_epochs + 1)

        state_dict = get_layer_proj_dict(args.index_path, args.base_dir, layer_i=n, matrix_type=args.matrix_type, model_variant=args.model_variant)
        if args.model_variant == "qwen3_5":
            key_prefix = f"model.language_model.layers.{n}.mlp.experts"
        elif args.model_variant == "gemma4":
            key_prefix = f"model.language_model.layers.{n}.experts"
        elif args.model_variant == "mixtral":
            key_prefix = f"model.layers.{n}.block_sparse_moe.experts"
        else:
            key_prefix = f"model.layers.{n}.mlp.experts"
        target_list = []
        for i in range(args.num_matrices):
            target_list.append(state_dict[f"{key_prefix}.{i}.{args.matrix_type}.weight"].to(torch.float16).to(device))
        target = torch.stack(target_list)
        target = orient_projection_for_training(target, args.matrix_type, args.model_variant)
        expected_shape = (args.num_matrices, args.rows_per_matrix, args.cols)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Unexpected {args.matrix_type} shape for layer {n}: "
                f"got {tuple(target.shape)}, expected {expected_shape} in training orientation"
            )
        global_target_std = target.std()

        if wandb_run is not None:
            wandb.log(
                {
                    "layer/current": n,
                    "layer/global_target_std": global_target_std.item(),
                },
                step=layer_step_base,
            )

        initial_A, initial_B, initial_W = None, None, None
        for group_i in range(num_group):
            gs = group_sizes[group_i]
            if gs == 0:
                continue
            W = target[group_offsets[group_i]:group_offsets[group_i+1]].view(-1, args.cols)
            W_float32 = W.to(torch.float32)
            U, S, Vt = torch.linalg.svd(W_float32, full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]

            group_A = U_k @ torch.diag(S_k)
            group_B = Vt_k.unsqueeze(0)
            group_W = torch.zeros(gs, num_group, device=device)
            group_W[:, group_i] = 1

            if initial_B is None:
                initial_B = group_B
            else:
                initial_B = torch.cat((initial_B, group_B), dim=0)

            for weight_i in range(gs):
                start_r = weight_i * args.rows_per_matrix
                end_r = (weight_i + 1) * args.rows_per_matrix
                single_A = group_A[start_r:end_r, :]
                single_W = group_W[weight_i:weight_i+1, :]

                initial_A = single_A.unsqueeze(0) if initial_A is None else torch.cat((initial_A, single_A.unsqueeze(0)), dim=0)
                initial_W = single_W if initial_W is None else torch.cat((initial_W, single_W), dim=0)

        model = MoBE(initial_A, initial_B, initial_W, activation=args.activation).to(device)
        del initial_A, initial_B, initial_W
        gc.collect()
        torch.cuda.empty_cache()
        if args.fast:
            model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        # Tensor buffers: avoid per-step .item() syncs
        _loss_buf      = torch.zeros(1, device=device)
        _real_loss_buf = torch.zeros(1, device=device)
        _real_mae_buf  = torch.zeros(1, device=device)
        best_real_loss_tensor = torch.full((1,), float('inf'), device=device)

        for epoch in range(args.num_epochs):
            need_log   = (wandb_run is not None and ((epoch + 1) % args.wandb_log_every == 0 or epoch == 0 or (epoch + 1) < 200))
            need_print = ((epoch + 1) % 200 == 0)
            need_stats = need_log or need_print
            need_diag  = need_stats and not args.fast  # grad/param norms

            _loss_buf.zero_()
            _real_loss_buf.zero_()
            if need_stats:
                _real_mae_buf.zero_()

            optimizer.zero_grad()
            for batch_idx in range(args.num_batches):
                start_idx = batch_idx * args.batch_size
                end_idx = min((batch_idx + 1) * args.batch_size, args.num_matrices)
                indices = torch.arange(start_idx, end_idx, device=device)
                if len(indices) == 0:
                    continue

                if args.fast:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(indices)
                    outputs = outputs.float()
                else:
                    outputs = model(indices)
                batch_target = target[indices]

                Z_scaled = batch_target / global_target_std
                Z_hat_unscaled = outputs * global_target_std

                loss      = F.mse_loss(outputs, Z_scaled.to(torch.float32))
                real_loss = F.mse_loss(Z_hat_unscaled, batch_target)

                (loss * len(indices) / args.num_matrices).backward()

                _loss_buf      += loss.detach()      * len(indices)
                _real_loss_buf += real_loss.detach() * len(indices)
                if need_stats:
                    real_mae = F.l1_loss(Z_hat_unscaled, batch_target)
                    _real_mae_buf += real_mae.detach() * len(indices)

            # Grad norms before optimizer step (only when needed)
            if need_diag:
                mean_grad_A = model.A_params.grad.detach().norm().item() if model.A_params.grad is not None else 0.0
                mean_grad_B = model.B_params.grad.detach().norm().item() if model.B_params.grad is not None else 0.0
                mean_grad_w = model.w_params.grad.detach().norm().item() if model.w_params.grad is not None else 0.0

            optimizer.step()

            # Best-model tracking via tensor comparison (one .item() sync only on new best)
            epoch_real_loss_t = _real_loss_buf / args.num_matrices
            if (epoch_real_loss_t < best_real_loss_tensor).item():
                best_real_loss_tensor.copy_(epoch_real_loss_t)
                best_real_loss = best_real_loss_tensor.item()
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

            if need_stats:
                epoch_loss      = (_loss_buf      / args.num_matrices).item()
                epoch_real_loss = epoch_real_loss_t.item()
                epoch_real_mae  = (_real_mae_buf  / args.num_matrices).item()

                if need_diag:
                    with torch.no_grad():
                        A_norm = torch.linalg.norm(model.A_params.detach().float(), dim=(1, 2))
                        B_norm = torch.linalg.norm(model.B_params.detach().float(), dim=(1, 2))
                        w_norm = torch.linalg.norm(model.w_params.detach().float(), dim=1)

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
                    }
                    if need_diag:
                        log_dict.update({
                            "grad_norm/A_mean": mean_grad_A,
                            "grad_norm/B_mean": mean_grad_B,
                            "grad_norm/w_mean": mean_grad_w,
                            "A_norm/min": A_norm.min().item(),
                            "A_norm/mean": A_norm.mean().item(),
                            "A_norm/max": A_norm.max().item(),
                            "A_norm/std": A_norm.std().item(),
                            "B_norm/min": B_norm.min().item(),
                            "B_norm/mean": B_norm.mean().item(),
                            "B_norm/max": B_norm.max().item(),
                            "B_norm/std": B_norm.std().item(),
                            "w_norm/min": w_norm.min().item(),
                            "w_norm/mean": w_norm.mean().item(),
                            "w_norm/max": w_norm.max().item(),
                            "w_norm/std": w_norm.std().item(),
                        })
                    wandb.log(log_dict, step=step)

                if need_print:
                    print(f"Epoch {epoch+1}, Scaled Loss: {epoch_loss:.10f}, Real MSE Loss: {epoch_real_loss:.10f} (best {best_real_loss:.10f} at {best_epoch + 1}), Real MAE Loss: {epoch_real_mae:.10f}")
                    if need_diag:
                        print(
                            "A Fro-norm [min/mean/max/std]: "
                            f"{A_norm.min().item():.4f} / {A_norm.mean().item():.4f} / "
                            f"{A_norm.max().item():.4f} / {A_norm.std().item():.4f}"
                        )
                        print(
                            "B slice Fro-norm [min/mean/max/std]: "
                            f"{B_norm.min().item():.4f} / {B_norm.mean().item():.4f} / "
                            f"{B_norm.max().item():.4f} / {B_norm.std().item():.4f}"
                        )
                        print(
                            "w Fro-norm [min/mean/max/std]: "
                            f"{w_norm.min().item():.4f} / {w_norm.mean().item():.4f} / "
                            f"{w_norm.max().item():.4f} / {w_norm.std().item():.4f}"
                        )
                        print(
                            "Grad L2-norm [A/B/w]: "
                            f"{mean_grad_A:.6f} / {mean_grad_B:.6f} / {mean_grad_w:.6f}"
                        )

        model.load_state_dict(best_model_state)

        model.eval()
        reconstructed_dict = {}
        with torch.no_grad():
            indices = torch.arange(0, args.num_matrices, device=device)
            outputs = model(indices)
            Z_hat_unscaled = outputs * global_target_std
            for weight_i in range(args.num_matrices):
                key = f'experts_{weight_i}_{args.matrix_type}_weight'
                reconstructed_dict[key] = orient_projection_for_checkpoint(
                    Z_hat_unscaled[weight_i], args.matrix_type, args.model_variant
                )

        output_path = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_weight.safetensors'
        save_file(reconstructed_dict, output_path)

        out_file = f'{args.save_path}/model_layers_{n}_mlp_{args.matrix_type}_WAB.pth'
        torch.save(model.state_dict(), out_file)
        print(f"Saved best model weights for layer {n} at {out_file} with loss {best_real_loss:.10f}")

        model_params = sum(p.numel() for p in model.parameters())
        compression_ratio = model_params / (args.num_matrices * args.rows_per_matrix * args.cols)
        print(f'Layer: {n}, Compression Ratio: {compression_ratio:.5f}')

        if wandb_run is not None:
            layer_summary = {
                f"layer/{n}/best_real_mse": best_real_loss,
                f"layer/{n}/best_epoch": best_epoch + 1,
                f"layer/{n}/compression_ratio": compression_ratio,
                f"layer/{n}/checkpoint_path": out_file,
                f"layer/{n}/reconstruction_path": output_path,
            }
            wandb.log(layer_summary, step=layer_step_base + args.num_epochs)
            wandb_run.summary[f"layer_{n}_best_real_mse"] = best_real_loss
            wandb_run.summary[f"layer_{n}_best_epoch"] = best_epoch + 1
            wandb_run.summary[f"layer_{n}_compression_ratio"] = compression_ratio
            if args.wandb_save_artifacts:
                artifact = wandb.Artifact(
                    name=f"layer-{n}-{args.matrix_type}",
                    type="train2-layer-output",
                    metadata={
                        "layer": n,
                        "matrix_type": args.matrix_type,
                        "best_real_mse": best_real_loss,
                        "best_epoch": best_epoch + 1,
                        "compression_ratio": compression_ratio,
                    },
                )
                artifact.add_file(output_path)
                artifact.add_file(out_file)
                wandb_run.log_artifact(artifact)

        del target, target_list, state_dict, model, optimizer, model_params
        if 'initial_A' in globals():
            del initial_A
        if 'initial_B' in globals():
            del initial_B
        if 'initial_W' in globals():
            del initial_W

        gc.collect()
        torch.cuda.empty_cache()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
