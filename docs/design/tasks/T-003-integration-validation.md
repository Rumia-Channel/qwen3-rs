# T-003: CLI, tokenizer, documentation, and reference validation

## Goal
The Qwen3.5-2B text path is usable from the CLI, terminates correctly, is documented honestly, and has reproducible local reference checks.

## Context
- Read first: `docs/design/design.md`, T-001/T-002 implementation, `qwen3-inference/src/generation.rs`, tokenizer/exporter/chat-template code, `qwen3-cli/src/main.rs`, `README.md`, local model/Transformers files.
- Invariants: no image API; existing Qwen3 commands remain valid; cache reset on rollover.

## Requirements
1. Route prompt ingestion through the prefill API and invoke cache reset on chat rollover/new context.
2. Stop Qwen3.5 on chat EOS 248046 and additional model EOS 248044, without changing existing Qwen3 stop semantics.
3. Ensure Qwen3.5 text/system/thinking templates match the local tokenizer template for the CLI-supported subset.
4. Update README support matrix, exact export/run examples, 4096 default, context memory warning, text-only/vision/MoE limitations, and reference provenance.
5. Add a local reference fixture generator/checker using the supplied venv/model where practical; keep large model outputs out of git. Compare tokenizer strings and deterministic short-step outputs, clearly separating BF16 and Q8 expectations.
6. Run and report workspace checks, real export, and a short greedy smoke generation. Do not claim numerical parity if it was not measured.

## Non-goals
- Tool-call API, multimodal messages, performance optimization, or support for other Qwen3.5 sizes.

## Acceptance
- Run all commands in design.md §Tests and acceptance; real export and smoke generation may be long but must either complete or have the concrete blocker recorded.

## Decision policy
- Decide yourself: wording and fixture output location.
- Escalate: changing CLI flags/defaults beyond Qwen3.5-specific context behavior or weakening correctness claims.
