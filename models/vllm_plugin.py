"""vLLM plugin: registers SLBF / Naive-MoE Mixtral model classes.

Loaded by vLLM's general plugin system in main and worker processes via the
'vllm.general_plugins' entry point declared in pyproject.toml.
"""


def register_models() -> None:
    from vllm import ModelRegistry
    # Lazy registration avoids importing version-sensitive custom layers when
    # this package is installed in a stock-model evaluation environment.
    ModelRegistry.register_model(
        "MixtralNaiveMoEForCausalLM",
        "models.vllm_mixtral_naive_moe:MixtralNaiveMoEForCausalLM",
    )
    ModelRegistry.register_model(
        "MixtralSLBFForCausalLM", "models.vllm_mixtral_slbf:MixtralSLBFForCausalLM",
    )
