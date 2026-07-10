# Trial-and-error journal

### A-001: Initial repository and upstream reconnaissance (2026-07-10)
- Tried: Mapped the existing Qwen3 exporter/inference path and attempted exact upstream Qwen3.5-2B research plus standard local HF/Transformers cache lookup.
- Expected: Obtain official config, modeling source, and tensor layout sufficient to finalize the v2 schema.
- Happened: The repository map confirmed a Qwen3-only fixed-order v1 format with hard-coded RoPE/RMS parameters. The research runtime had no web tools, and the standard local Qwen3.5/Transformers paths checked were absent.
- Learned: Qwen3.5 must not be implemented from assumptions; a pinned official artifact/source snapshot is required.
- Design impact: updated design.md §Constraints, §Chosen approach, and §Open questions.

### A-002: Implement checkpoint v2 and hybrid inference (2026-07-10)
- Tried: Added nested Qwen3.5 config parsing, architecture ID 3, a v2 header, language-only tensor export, hybrid full-attention/DeltaNet inference, distinct norm conventions, FP32 recurrent caches, selected-row embeddings, and final-only-logits prefill.
- Expected: A compiling implementation whose stream order and token recurrence match the pinned local Transformers source.
- Happened: Multiple delegated implementation passes required static repair of config field names, tensor ordering, gated norm shape, residual preservation, q/gate layout, partial RoPE pairing, convolution history, and prefill state. Fresh static reviewers found no remaining offset/equation blocker for the exact 2B config.
- Learned: The fixed-order format must group v2 FP32 Delta tensors before Q8 tensors, and absent architecture-specific norms require fixed-width placeholder slots for deterministic offsets.
- Design impact: implemented design.md §Exact architecture, §Checkpoint v2, and §Runtime and cache lifecycle.

### A-003: Integrate generation, chat, and documentation (2026-07-10)
- Tried: Routed initial prompts through `reset_cache` and `prefill`, added cache reset on chat rollover, consumed chat stop tokens into the cache, handled both Qwen3.5 EOS IDs, fixed CLI context typing, updated thinking templates, and documented limits.
- Expected: Workspace checks plus a real export and short deterministic smoke generation.
- Happened: After configuring Scoop Git Bash and restarting Pi, `cargo fmt --check`, `cargo check --workspace`, warning-free Clippy, 65 unit tests plus one doctest, a real 1.9 GiB Q8 export, and a short greedy inference smoke test all passed. The smoke prompt `Hello` produced `, I am a 20` at 11.76 tokens/s with an eight-token context. Transformers numerical comparison could not run because the local venv lacks `torch`.
- Learned: The original `.gitignore` pattern `Qwen3-*` also ignored the `qwen3-export` and `qwen3-inference` crate trees on case-insensitive Windows, including both new `qwen3_5.rs` files; it was narrowed to root model directories before completion.
- Design impact: implementation status updated; numerical parity remains the sole major acceptance gap.
