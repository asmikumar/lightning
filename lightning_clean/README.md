`lightning_clean` is a minimal snapshot of the local files used by `scripts/run_grpo_training_quick_full_prompt.lsf`.

Included:
- `scripts/run_grpo_training_quick_full_prompt.lsf`
- `scripts/train_prompt_optimization_verl_direct_edits-4.py`

Not copied because they are external runtime dependencies rather than local files under `lightning/`:
- `/u/asmi/dsl-agent/dspy_tutorial/wiki2017`
- `/u/asmi/dsl-agent/dspy_tutorial/mellea_tutorial/programs/dspy_optimized/best_program.json`
- Python packages from the `py311_base` environment
- Hugging Face dataset `vincentkoc/hover-parquet`
- Modules resolved via sibling paths such as `dspy_tutorial` and `mellea_hover_baseline`

Notes:
- The copied LSF script still launches `scripts/train_prompt_optimization_verl_direct_edits-4.py`.
- The copied Python script still expects the same absolute paths and environment variables as the original job.
