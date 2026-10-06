"""Launch one SLBF training job per layer/projection from a YAML recipe.

No torch import or GPU allocation is needed for --dry_run. Relative paths are
resolved against the release root, not the caller's working directory.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESERVED = {"index_path", "base_dir", "save_path", "start_layer", "end_layer",
            "matrix_type", "wandb", "wandb_mode", "wandb_run_name"}
PROJECTIONS = {"gate_proj", "up_proj"}


def trainer_options():
    tree = ast.parse((ROOT / "train_lrbase.py").read_text())
    return {node.args[0].value[2:] for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument" and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("--")}


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Training settings must be a mapping")
    invalid = set(settings) - (trainer_options() - RESERVED)
    if invalid:
        raise ValueError(f"Unknown/reserved trainer settings: {sorted(invalid)}")
    if any(not isinstance(value, (str, int, float, bool))
           for value in settings.values()):
        raise ValueError("Training settings must be scalar values")


def load_recipe(path):
    with open(path) as handle:
        recipe = yaml.safe_load(handle)
    if not isinstance(recipe, dict) or set(recipe) != {"model", "defaults", "overrides"}:
        raise ValueError("Recipe must contain model, defaults, and overrides")
    model = recipe["model"]
    if not isinstance(model, dict) or set(model) != {
        "base_dir", "save_path", "start_layer", "end_layer", "projections"
    }:
        raise ValueError("Unexpected model settings")
    start, end = model["start_layer"], model["end_layer"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= 32:
        raise ValueError("Mixtral layer range must satisfy 0 <= start < end <= 32")
    if not isinstance(model["projections"], list) or not model["projections"]:
        raise ValueError("projections must be a nonempty list")
    if not set(model["projections"]) <= PROJECTIONS:
        raise ValueError("This release compresses gate/up only")
    validate_settings(recipe["defaults"])
    if recipe["defaults"].get("model_variant") != "mixtral":
        raise ValueError("This release supports the mixtral variant only")
    if not isinstance(recipe["overrides"], dict):
        raise ValueError("overrides must be a mapping")
    for layer, overrides in recipe["overrides"].items():
        if type(layer) is not int or not start <= layer < end:
            raise ValueError(f"Override layer out of range: {layer}")
        if not isinstance(overrides, dict) or not overrides or not set(overrides) <= set(model["projections"]):
            raise ValueError(f"Invalid projection override for layer {layer}")
        for settings in overrides.values():
            validate_settings(settings)
            if "model_variant" in settings and settings["model_variant"] != "mixtral":
                raise ValueError("Overrides cannot change architecture")
    return recipe


def plan_jobs(recipe, start=None, end=None, projections=None, only_overrides=False,
              ignore_overrides=False):
    model = recipe["model"]
    start = model["start_layer"] if start is None else start
    end = model["end_layer"] if end is None else end
    projections = model["projections"] if projections is None else projections
    if not model["start_layer"] <= start < end <= model["end_layer"]:
        raise ValueError("Requested layers are outside the recipe range")
    if not set(projections) <= set(model["projections"]):
        raise ValueError("Requested projections are outside the recipe")
    if only_overrides and ignore_overrides:
        raise ValueError("only_overrides and ignore_overrides are incompatible")
    jobs = []
    for layer in range(start, end):
        for projection in projections:
            override = recipe["overrides"].get(layer, {}).get(projection)
            if only_overrides and override is None:
                continue
            settings = dict(recipe["defaults"])
            if not ignore_overrides and override is not None:
                settings.update(override)
            if settings.get("activation", "silu") != "silu" or settings.get("norm_mode", "global_std") != "global_std":
                raise ValueError("The released pack/runtime path requires SiLU and global_std")
            if any(settings.get(key, False) for key in
                   ("uv_normalize", "no_w_softmax", "w_scale", "b_normalize")):
                raise ValueError("Experimental normalization/mixing flags are unsupported by the pack/runtime path")
            jobs.append((layer, projection, settings))
    if not jobs:
        raise ValueError("No training jobs selected")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mixtral_slbf_k832.yaml")
    parser.add_argument("--base_dir", type=Path)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--start_layer", type=int)
    parser.add_argument("--end_layer", type=int, help="Exclusive upper bound")
    parser.add_argument("--projections", nargs="+", choices=sorted(PROJECTIONS))
    parser.add_argument("--only_overrides", action="store_true")
    parser.add_argument("--ignore_overrides", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    args = parser.parse_args()
    if args.skip_existing and args.overwrite:
        parser.error("--skip_existing and --overwrite are incompatible")
    recipe = load_recipe(args.config)
    jobs = plan_jobs(recipe, args.start_layer, args.end_layer, args.projections,
                     args.only_overrides, args.ignore_overrides)
    base = args.base_dir or Path(recipe["model"]["base_dir"])
    output = args.save_path or Path(recipe["model"]["save_path"])
    base = (ROOT / base).resolve()
    output = (ROOT / output).resolve()
    if not args.dry_run and not (base / "model.safetensors.index.json").is_file():
        parser.error(f"Missing base-model safetensors index: {base}")
    pending = []
    for layer, projection, settings in jobs:
        checkpoint = output / f"model_layers_{layer}_mlp_{projection}_WAB.pth"
        if checkpoint.exists():
            if args.skip_existing:
                continue
            if not args.overwrite:
                parser.error(f"Refusing to overwrite {checkpoint}; use --skip_existing or --overwrite")
        pending.append((layer, projection, settings))
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    for layer, projection, settings in pending:
        command = [sys.executable, str(ROOT / "train_lrbase.py"),
                   "--index_path", str(base / "model.safetensors.index.json"),
                   "--base_dir", str(base), "--save_path", str(output),
                   "--start_layer", str(layer), "--end_layer", str(layer + 1),
                   "--matrix_type", projection, "--wandb_mode", args.wandb_mode]
        for key, value in settings.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            else:
                command.extend([f"--{key}", str(value)])
        if args.wandb_mode != "disabled":
            command += ["--wandb", "--wandb_run_name", f"mixtral_slbf_L{layer}_{projection}"]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            # Record resolved settings per artifact, including source config.
            manifest = {"layer": layer, "projection": projection, "settings": settings,
                        "base_dir": str(base), "config": str(args.config.resolve()),
                        "command": command}
            manifest_path = output / f"model_layers_{layer}_mlp_{projection}_settings.json"
            with manifest_path.open("w") as handle:
                json.dump(manifest, handle, indent=2)
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
