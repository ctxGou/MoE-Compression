# TD-MoE: Tensor Decomposition for MoE Models

This repository contains the official implementation for our paper **"TD-MoE: Cross-Expert Decomposition for MoE Models"**.

## Installation

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate MoeComp
```     

### Key Dependencies

- Python 3.11+
- PyTorch 2.0+
- Transformers 4.30+
- TensorLy
- NumPy, SciPy

## Quick Start

### 1. Basic Compression

Compress a Mixtral model with 20% compression ratio:

```bash
bash tucker_mixtral.sh
```

### 2. Custom Compression

```bash
python src/run_tucker.py \
    --model_path /path/to/model \
    --save_path ./results \
    --ratio 0.2 \
    --layers_to_compress 3 5 6 7 9 12 23 24 25 \
    --cluster_type global \
    --whiten_type both \
    --decomposition_method hosvd
```

### 3. Evaluation

```bash
python src/run_evaluation.py \
    --model_path /path/to/model \
    --save_path ./results \
    --ratio 0.2 \
    --layers_to_compress 3 5 6 7 9 12 23 24 25 \
    --cluster_type global \
    --eval_tasks winogrande piqa arc_easy arc_challenge
```

## Directory Structure

```
TD-MoE/
├── src/                                    # Core implementation
│   ├── config.py                          # Model configurations for different MoE architectures
│   ├── data_collection.py                 # Activation and gradient collection via hooks
│   ├── tucker_decomposition.py            # Core Tucker decomposition algorithms
│   ├── layer_selection.py                 # Layer sensitivity analysis
│   ├── run_tucker.py                      # Main compression pipeline
│   ├── run_evaluation.py                  # Evaluation pipeline
│   ├── evaluator.py                       # Task evaluation utilities
│   └── components/                        # Model-specific MoE implementations
│       ├── base_moe.py                    # Base Tucker-decomposed MoE class
│       ├── tucker_mixtral.py              # Mixtral-specific implementation
│       └── tucker_phi.py                  # Phi-3.5-MoE-specific implementation
│
├── scripts/                               # Experiment scripts
│   ├── tucker_mixtral.sh                  # Mixtral compression script
│   └── tucker_phi.sh                      # Phi-3.5-MoE compression script
│
├── lm-evaluation-harness/                  # Evaluation framework
│   └── ...                               # Modified lm-eval for MoE evaluation
│
├── results/                               # Experimental outputs
│   ├── decomposition_results/             # Compressed model components  
│   │   └── {model_name}/
│   │       ├── global/                    # Global compression results
│   │       ├── group/                     # Group-based compression results
│   │       └── adaptive/                  # Adaptive compression results
│   ├── evaluation_results/                # Task evaluation results
│   └── covariances/                       # Cached covariance matrices
│
├── rank_allocation/                       # Rank allocation analysis
│   ├── outputs.csv                        # Compression ratio analysis
│   └── results/                           # Visualization outputs
│
└── environment.yml                        # Conda environment specification
```

## Methodology

### Core Components

1. **Covariance Collection**: Collect activation and gradient statistics using forward hooks
2. **Whitening Transformation**: Apply Cholesky decomposition for input/output whitening  
3. **Tucker Decomposition**: Decompose whitened tensors using HOSVD or ALS algorithms
4. **Model Reconstruction**: Rebuild compressed MoE layers with Tucker factors

### Compression Modes

- **Global**: All experts share the same decomposition basis
- **Group**: Experts are clustered and decomposed by groups  
- **Mixed**: Combine global and group strategies for different layers
- **Adaptive**: Use layer sensitivity to determine compression ratios

### Whitening Strategies

- `input`: Whiten input activations only
- `output`: Whiten output gradients only  
- `both`: Apply both input and output whitening
- `none`: No whitening (baseline Tucker decomposition)

## Experiments

### Evaluation Tasks  

- **Language Modeling**: WikiText2, PTB perplexity
- **Commonsense Reasoning**: PIQA, WinoGrande, OpenBookQA
- **Reading Comprehension**: ARC-Easy, ARC-Challenge


## Configuration

Key parameters in compression scripts:

- `--ratio`: Global compression ratio (0.1-0.8)
- `--layers_to_compress`: List of target layer indices  
- `--cluster_type`: Compression mode (global/group/mixed)
- `--whiten_type`: Whitening strategy (input/output/both/none)
- `--decomposition_method`: Tucker algorithm (hosvd/als)
- `--whitening_nsamples`: Calibration samples for covariance estimation

