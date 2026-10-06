We provide code, paper, and poster of  ***Shared Low-rank Basis Factorization for Data-free Mixture-of-Experts Compression***.

This code trains and evaluates SLBF on Mixtral-8x7B-v0.1. It includes
gauge-fixed checkpoint conversion, a vLLM runtime plugin, and a MoBE baseline.

## Set up

Run the commands from the directory that contains this README.

```bash
conda create -n slbf python=3.11 -y
conda activate slbf
pip install -r requirements.txt
```

## Train

Set `BASE_MODEL` to a local Mixtral checkpoint directory with safetensors weights.

```bash
BASE_MODEL=/path/to/Mixtral-8x7B-v0.1
python scripts/train/from_yaml.py --base_dir "$BASE_MODEL" --gpu 0
```

The [training configuration](configs/mixtral_slbf_k832.yaml) sets the common
parameters and the layer-specific learning rates. The recipe uses eight
bases of rank 832. It compresses the gate and up projections and keeps the
down projections unchanged.

The trainer saves factor checkpoints in `results/wab/mixtral/slbf_k832_final`.
Add `--dry_run` to show the commands. Add `--skip_existing` to resume training.

## Evaluate

Create an evaluation environment. The requirements specify lm-eval 0.4.11,
vLLM 0.19.1, and Transformers 5.6.2.

```bash
conda create -n slbf-eval python=3.12 -y
conda activate slbf-eval
pip install -r requirements-eval.txt
```

Build a standard Mixtral checkpoint from the trained factors. Use a new
output directory.

```bash
python get_hf_model_from_wab.py \
  --base_model "$BASE_MODEL" --mobe_dir results/wab/mixtral/slbf_k832_final \
  --save_dir results/materialized/mixtral_slbf \
  --start_layer 0 --end_layer 32 --num_experts 8 \
  --model_variant mixtral --gauge_fix --skip_projections down_proj \
  --required_projections gate_proj up_proj

bash scripts/eval/mixtral/run_eval.sh results/materialized/mixtral_slbf slbf 0,1,2,3
```

The evaluator uses the supplied [task templates](scripts/eval/mixtral/tasks/README.md).
It runs eight zero-shot tasks: ARC-Challenge, ARC-Easy, HellaSwag,
OpenBookQA, RTE, WinoGrande, PIQA, and Wikitext. It saves results in
`results/eval` and logs in `logs`.

## Use a compact checkpoint

Pack the trained factors into the gauge-fixed format. Use a new output directory.

```bash
conda activate slbf
python scripts/build/mixtral/pack_gauge_fixed.py \
  --base_model "$BASE_MODEL" --wab_dir results/wab/mixtral/slbf_k832_final \
  --save_dir results/compact/mixtral_slbf
```

Install the runtime plugin in a separate environment. It uses vLLM 0.17.1.

```bash
conda create -n slbf-runtime python=3.11 -y
conda activate slbf-runtime
pip install -r requirements-runtime.txt
pip install --no-deps -e .

CUDA_VISIBLE_DEVICES=0,1 python scripts/bench/mixtral_smoke_test.py \
  --model results/compact/mixtral_slbf --pp 2
```

The runtime reconstructs weights during inference. The
[benchmark script](scripts/bench/mixtral_throughput.py) measures throughput
and GPU memory use. The [unpack script](scripts/build/mixtral/unpack_to_materialized.py)
converts a compact checkpoint into standard Mixtral weights.

## Run tests

Run the CPU tests in the training environment.

```bash
conda activate slbf
python -m unittest discover -s tests -v
```

## Acknowledgement

This repository builds upon [MoBE](https://github.com/inclusionAI/MoBE). We thank the authors for releasing their code.

## Citation

<!-- TODO: Add the paper citation. -->

