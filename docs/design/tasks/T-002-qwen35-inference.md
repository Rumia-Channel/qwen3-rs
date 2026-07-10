# T-002: Qwen3.5 hybrid text inference

## Goal
A v2 Qwen3.5 checkpoint can be loaded and run token-by-token using the official dense 2B text equations and correct hybrid caches.

## Context
- Read first: `docs/design/design.md`, T-001 implementation, official `modeling_qwen3_5.py` functions/classes for RMSNorm, GatedDeltaNet, Attention, DecoderLayer, TextModel, and existing inference tensor/layer/model code.
- Invariants: preserve Qwen3 output/API behavior; FP32 recurrence semantics; standard and gated norms differ; all caches reset cleanly.

## Requirements
1. Add Qwen3.5 model dispatch and load the v2 stream exactly with shape/bounds context on every failure.
2. Implement quantized selected-row embedding lookup without full-table FP32 dequantization.
3. Implement standard `(1+w)` RMSNorm, direct-weight gated RMSNorm+SiLU, partial text RoPE, gated GQA full attention, SwiGLU, and residual ordering exactly as official source.
4. Implement depthwise causal convolution and single-token FP32 Gated Delta recurrence exactly, including L2 epsilon, Q scale, beta/g computation, state layout, and per-layer state.
5. Implement separate per-layer full KV or linear conv/recurrent caches and `reset_cache`; reject `pos` misuse rather than silently corrupt state.
6. Add prompt prefill support that uses exact token recurrence but computes logits only for the final prompt token; retain backward compatibility for callers of `forward`.
7. Default Qwen3.5 runtime context to 4096 if unspecified, bound overrides to model max, and use checked allocations.
8. Add deterministic unit tests for primitives, cache update/reset, short sequence equivalence, and malformed checkpoint behavior.

## Non-goals
- No chunk-size-64 bulk Delta kernel, vision, MoE, training, or GPU path.

## Acceptance
- Run: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.

## Decision policy
- Decide yourself: module split and private buffer reuse.
- Escalate: numerical deviations from pinned official equations, extra dependencies, public API breakage without a compatibility default, or memory representation changes from design.md.
