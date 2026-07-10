# T-001: Qwen3.5 config, exporter, and checkpoint v2

## Goal
The exporter recognizes the official Qwen3.5-2B wrapper, extracts its dense text config, writes only language-model weights in a validated v2 stream, and leaves Qwen3 v1 behavior intact.

## Context
- Read first: `docs/design/design.md`, `sample/Qwen/Qwen3.5-2B/config.json`, `sample/.venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`, `qwen3-export/src/config_loader.rs`, `qwen3-export/src/model_exporter.rs`, `qwen3-export/src/models/*`, `qwen3-inference/src/configuration.rs`.
- Invariants: architecture ID 2 remains Llama; Qwen3 v1 bytes/reader remain supported; visual and MTP are never exported.

## Requirements
1. Add architecture ID 3 for Qwen3.5 dense text and parse the nested official `text_config` strictly, validating the exact supported feature set while deriving/storing all v2 metadata from design.md.
2. Add a Qwen3.5 exporter architecture using `model.language_model.*`, hybrid layer-specific names, tied embeddings, and explicit FP32-vs-Q8 tensor classification/order.
3. Accept and skip only known `model.visual.*`/`mtp.*`; detect missing, duplicate, unknown `model.language_model.*`, wrong shapes, unsupported MoE/bias/activation/untied settings with actionable errors.
4. Implement a 256-byte v2 header and version-aware reader while preserving v1. Use checked conversions/arithmetic.
5. Parse tokenizer chat EOS from `tokenizer_config.json` rather than blindly using nested model EOS for Qwen3.5.
6. Add focused tests for official config values/layer mask, architecture dispatch, v1/v2 reading, tensor order/classification, and rejection paths.

## Non-goals
- No inference kernels, vision export, MoE, or chunked prefill.
- Do not quantize `A_log`, `dt_bias`, convolution, or norm tensors.

## Acceptance
- Run: `cargo fmt --all -- --check`, `cargo test -p qwen3-export`, `cargo test -p qwen3-inference configuration`.

## Decision policy
- Decide yourself: private helper names and internal test fixture organization.
- Escalate: changing architecture IDs, header size, tensor order/storage class, supported config semantics, or adding dependencies.
