import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_module("slbf_trainer", "train_lrbase.py")
packer = load_module("slbf_packer", "scripts/build/mixtral/pack_gauge_fixed.py")
unpacker = load_module("slbf_unpacker", "scripts/build/mixtral/unpack_to_materialized.py")
launcher = load_module("slbf_launcher", "scripts/train/from_yaml.py")


class MixtralReleaseTests(unittest.TestCase):
    def test_slbf_equation_and_gradients(self):
        torch.manual_seed(10)
        A, U, V, W = [torch.randn(shape, dtype=torch.float64) for shape in
                      [(2, 4, 4), (2, 4, 2), (2, 6, 2), (2, 2)]]
        model = trainer.LRBMoBE(A, U, V, W)
        indices = torch.tensor([1, 0])
        bases = U @ V.transpose(-1, -2)
        expected = A[indices] @ F.silu(torch.einsum("nm,mrd->nrd", W[indices].softmax(-1), bases))
        torch.testing.assert_close(model(indices), expected)
        model(indices).square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_gauge_fix_identity_and_pivot_partition(self):
        torch.manual_seed(11)
        U = torch.randn(3, 5, 2, dtype=torch.float64)
        V = torch.randn(3, 7, 2, dtype=torch.float64)
        free, right, pivots, rest = packer.gauge_fix(U, V, torch.float64)
        left = torch.empty_like(U)
        for basis in range(3):
            left[basis, pivots[basis].long()] = torch.eye(2, dtype=torch.float64)
            left[basis, rest[basis].long()] = free[basis]
            self.assertEqual(sorted(torch.cat([pivots[basis], rest[basis]]).tolist()), list(range(5)))
        torch.testing.assert_close(left @ right.transpose(-1, -2), U @ V.transpose(-1, -2), atol=1e-10, rtol=1e-10)

    def test_yaml_selects_exact_six_overrides(self):
        recipe = launcher.load_recipe(ROOT / "configs/mixtral_slbf_k832.yaml")
        jobs = launcher.plan_jobs(recipe)
        self.assertEqual(len(jobs), 64)
        special = {(layer, projection) for layer, projection, settings in jobs
                   if settings["uv_lr_scale"] == 0.05}
        self.assertEqual(special, {(12, "up_proj"), (14, "gate_proj"),
                                  (14, "up_proj"), (17, "gate_proj"),
                                  (29, "gate_proj"), (31, "up_proj")})
        self.assertEqual(len(launcher.plan_jobs(recipe, only_overrides=True)), 6)
        self.assertTrue(all(settings["uv_lr_scale"] == 0.1 for _, _, settings in
                            launcher.plan_jobs(recipe, ignore_overrides=True)))
        with self.assertRaises(ValueError):
            launcher.plan_jobs(recipe, start=31, end=33)
        with self.assertRaises(ValueError):
            launcher.validate_settings({"unknown_option": 1})

    def test_unpack_reconstruction_matches_explicit_factors(self):
        torch.manual_seed(13)
        U, V = torch.randn(2, 4, 2), torch.randn(2, 6, 2)
        free, right, pivots, rest = packer.gauge_fix(U, V)
        A, W = torch.randn(2, 4, 4).bfloat16(), torch.randn(2, 2).bfloat16()
        state = {"U_free_gate": free, "V_hat_gate": right,
                 "pivots_gate": pivots, "rest_gate": rest, "A_gate": A, "W_gate": W}
        full_left = torch.empty(2, 4, 2)
        for basis in range(2):
            full_left[basis, pivots[basis].long()] = torch.eye(2)
            full_left[basis, rest[basis].long()] = free[basis].float()
        bases = full_left @ right.float().transpose(-1, -2)
        mixed = torch.einsum("nm,mrd->nrd", W.float().softmax(-1), bases)
        expected = (A.float() @ F.silu(mixed)).transpose(-1, -2).contiguous().bfloat16()
        actual = unpacker.reconstruct_layer(state, "gate", 4, 6, 2, 2, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_lazy_plugin_registration(self):
        registry = types.SimpleNamespace(register_model=unittest.mock.Mock())
        with patch.dict(sys.modules, {"vllm": types.SimpleNamespace(ModelRegistry=registry)}):
            plugin = load_module("slbf_plugin", "models/vllm_plugin.py")
            plugin.register_models()
        self.assertEqual(registry.register_model.call_count, 2)
        for call in registry.register_model.call_args_list:
            self.assertIsInstance(call.args[1], str)

    def test_frozen_task_templates_match_result_log(self):
        tasks_root = ROOT / "scripts/eval/mixtral/tasks"
        reference = json.loads((tasks_root.parent / "task_reference.json").read_text())
        class TaskLoader(yaml.SafeLoader):
            pass
        TaskLoader.add_constructor("!function", lambda loader, node: ("function", loader.construct_scalar(node)))

        def read_task(path):
            with path.open() as handle:
                config = yaml.load(handle, Loader=TaskLoader)
            if "include" in config:
                inherited = read_task(path.parent / config.pop("include"))
                inherited.update(config)
                config = inherited
            return config

        names = set()
        for path in tasks_root.rglob("*.yaml"):
            config = read_task(path)
            name = config["task"]
            names.add(name)
            recorded = reference["task_configs"][name]
            for field in ("dataset_path", "dataset_name", "output_type", "training_split",
                          "validation_split", "test_split", "doc_to_text", "doc_to_target",
                          "doc_to_choice", "process_docs", "process_results"):
                actual, expected = config.get(field), recorded.get(field)
                if isinstance(actual, tuple) and actual[0] == "function":
                    module, function = actual[1].rsplit(".", 1)
                    source = (path.parent / f"{module}.py").read_text()
                    tree = ast.parse(source)
                    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
                    self.assertEqual(ast.get_source_segment(source, node).strip(), expected.strip(), (name, field))
                else:
                    self.assertEqual(actual, expected, (name, field))
            self.assertEqual(recorded["num_fewshot"], 0)
        self.assertEqual(names, set(reference["task_configs"]))

    def test_toy_checkpoint_pack_unpack_and_output_guards(self):
        torch.manual_seed(12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, wab_dir, compact, dense = [root / name for name in ("base", "wab", "compact", "dense")]
            base.mkdir()
            wab_dir.mkdir()
            config = {"model_type": "mixtral", "architectures": ["MixtralForCausalLM"],
                      "num_hidden_layers": 1, "num_local_experts": 2,
                      "num_experts_per_tok": 1, "hidden_size": 4, "intermediate_size": 6}
            (base / "config.json").write_text(json.dumps(config))
            prefix = "model.layers.0.block_sparse_moe"
            original = {"model.embed_tokens.weight": torch.randn(10, 4).bfloat16(),
                        f"{prefix}.gate.weight": torch.randn(2, 4).bfloat16()}
            for expert in range(2):
                for weight, shape in [("w1", (6, 4)), ("w2", (4, 6)), ("w3", (6, 4))]:
                    original[f"{prefix}.experts.{expert}.{weight}.weight"] = torch.randn(shape).bfloat16()
            save_file(original, base / "model.safetensors")
            (base / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {key: "model.safetensors" for key in original}}))
            for projection in ("gate_proj", "up_proj"):
                factors = {name: torch.randn(shape) for name, shape in
                           [("A_params", (2, 4, 4)), ("U_params", (2, 4, 2)),
                            ("V_params", (2, 6, 2)), ("w_params", (2, 2))]}
                torch.save(factors, wab_dir / f"model_layers_0_mlp_{projection}_WAB.pth")
            pack_command = [sys.executable, str(ROOT / "scripts/build/mixtral/pack_gauge_fixed.py"),
                            "--base_model", str(base), "--wab_dir", str(wab_dir),
                            "--save_dir", str(compact), "--end_layer", "1", "--num_experts", "2",
                            "--num_B", "2", "--rank_per_basis", "2"]
            subprocess.run(pack_command, check=True, capture_output=True, text=True)
            unpack_command = [sys.executable, str(ROOT / "scripts/build/mixtral/unpack_to_materialized.py"),
                              "--slbf_dir", str(compact), "--save_dir", str(dense)]
            subprocess.run(unpack_command, check=True, capture_output=True, text=True)
            index = json.loads((dense / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(index), set(original))
            for key, filename in index.items():
                with safe_open(dense / filename, framework="pt") as handle:
                    value = handle.get_tensor(key)
                self.assertEqual(value.shape, original[key].shape)
                self.assertTrue(torch.isfinite(value).all())
                if ".w1." not in key and ".w3." not in key:
                    torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertEqual(json.loads((dense / "config.json").read_text())["architectures"], ["MixtralForCausalLM"])
            self.assertNotEqual(subprocess.run(pack_command, capture_output=True).returncode, 0)
            self.assertNotEqual(subprocess.run(unpack_command, capture_output=True).returncode, 0)
            invalid = pack_command.copy()
            invalid[invalid.index("--save_dir") + 1] = str(root / "partial")
            invalid += ["--start_layer", "1"]
            self.assertNotEqual(subprocess.run(invalid, capture_output=True).returncode, 0)
            self.assertFalse((root / "partial").exists())


if __name__ == "__main__":
    unittest.main()
