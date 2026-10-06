# Frozen Mixtral evaluation tasks

These task YAMLs and helpers are from the local lm-eval 0.4.11 evaluation
environment, checked against the recorded k832 run from 2026-05-11.
They are loaded explicitly through `--include_path`; installing stock lm-eval
alone is not treated as sufficient to reproduce the task templates.

`../task_reference.json` records the evaluated task configurations (including
serialized helper-function source), omitting machine-specific model metadata
and redundant few-shot configuration. It is a reference, not executable code.
The original result path is included for provenance; no weights or samples
are bundled. `tests/test_mixtral.py` checks the frozen fields and function
sources against this reference without downloading datasets.

Notable preserved settings include PIQA's `Question: ...\nAnswer:` prompt,
ARC's question/answer prompt, RTE's True/False choices, the explicit dataset
repository mappings, Winogrande's likelihood-continuation conversion, and
Wikitext's detokenization/word-count helper. All eight tasks are zero-shot;
none uses a chat template. The CoT/code-generation template modifications
for other model families are outside this Mixtral-only release.

Third-party task files/helpers retain the upstream MIT license in
`LICENSE.lm-eval.md`.
