# Changelog

## v0.1.0 — 2026-05-14

Initial release.

### Added
- `sdft_loss`, `sft_loss`, `_logprobs_at_positions` — core loss functions
- `sample_rollout` — detached student rollout via `mlx_lm.stream_generate`
- `EMATeacher` — frozen EMA teacher wrapper with `.update()` and `.save()`
- `SDFTTrainer` — high-level training API with `from_pretrained`, `fit`, `save`, `push_to_hub`
- `_make_loss_fn` hook for subclass customisation (e.g. Borda-SDFT)
- `tokenize_pairs`, `load_ultrachat`, `infinite_batches` — data utilities
- `push_to_hub`, `generate_model_card` — HuggingFace Hub integration
- Full test suite: losses, EMA math, rollout shapes, 100-step trainer smoke test
- Examples: minimal, full Qwen recipe, SFT vs SDFT comparison
- Docs: algorithm derivation, design rationale
