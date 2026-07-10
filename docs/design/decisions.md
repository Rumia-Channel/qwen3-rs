# Decision log

### D-001: Scope Qwen3.5 support to dense 2B text generation (2026-07-10)
- Context: Qwen3.5 includes model variants and potentially multimodal/MoE paths beyond the requested use case.
- Options: A) support the full family; B) support only `Qwen/Qwen3.5-2B` text generation.
- Decision: B. Vision and MoE are explicitly unsupported.
- Because: This matches the requested scope and keeps correctness/reference validation tractable.
- Council input: Pending design review after official upstream sources are available.
- Revisit if: The user requests another Qwen3.5 variant.

### D-002: Use a distinct architecture and versioned checkpoint path (2026-07-10)
- Context: The existing format is an unnamed fixed-order Qwen3 v1 stream and omits multiple forward-semantic parameters.
- Options: A) alias Qwen3.5 to Qwen3; B) introduce distinct architecture dispatch and v2 metadata.
- Decision: B, subject to final schema details from the official implementation.
- Because: It preserves Qwen3 compatibility and prevents silent tensor/semantic mismatch.
- Council input: Implemented as architecture_id=3, version=2.
- Revisit if: Official source proves the text model exactly compatible with the current equations, metadata, and tensor order.

### D-003: Accept the multimodal wrapper but export only its language model (2026-07-10)
- Context: The official 2B artifact is `Qwen3_5ForConditionalGeneration` and contains vision and MTP tensors despite the text-only product scope.
- Options: A) reject the wrapper; B) allowlist the wrapper and consume only `model.language_model.*`.
- Decision: B. Known `model.visual.*` and `mtp.*` tensors are ignored; no vision API is exposed.
- Because: Rejecting all vision tensors would make the requested official artifact unusable.
- Council input: Product and robustness reviewers independently identified this as a blocker in the initial draft.
- Revisit if: Qwen publishes a separate text-only 2B checkpoint with different prefixes.

### D-004: Correct recurrent prefill before chunk optimization (2026-07-10)
- Context: Official Transformers uses a chunk-size-64 DeltaNet prefill, while the existing runtime API is token-oriented.
- Options: A) require chunked prefill initially; B) use exact token recurrence and skip the LM head except for the final prompt token.
- Decision: B.
- Because: It preserves the official state transition with substantially lower implementation risk; chunking is a performance follow-up.
- Council input: Performance review preferred chunking immediately; simplicity and robustness favored a smaller correctness-first kernel. Final-only logits address the largest avoidable prompt cost.
- Revisit if: Measured prompt throughput is not usable.

### D-005: Default Qwen3.5 runtime context to 4096 (2026-07-10)
- Context: The model advertises 262144 positions, but six FP32 full-attention caches would require roughly 6 GiB at that length.
- Options: A) allocate model maximum by default; B) default to 4096 and permit a bounded explicit override.
- Decision: B.
- Because: The 4096 cache is roughly 96 MiB for full attention plus about 20 MiB for linear state, avoiding a surprising multi-gigabyte allocation.
- Council input: Performance and robustness reviews agreed the model maximum must not be the default allocation.
- Revisit if: Cache precision/layout changes or dynamic allocation is implemented.

### D-003: Prefill uses final-only LM head (2026-07-11)
- Context: The prefill API processes all prompt tokens but computes logits only for the final token, saving the LM head for all but the last prompt token.
- Options: A) token-by-token forward for all tokens (old behavior); B) prefill with final-only LM head.
- Decision: B. For Qwen3.5, `forward_internal` with `compute_logits=false` skips the 248k-way LM head for non-final prompt tokens. Qwen3 uses the default trait implementation which still computes logits each time.
- Because: Avoids unnecessary compute on the largest single matrix multiplication, particularly impactful at large vocab sizes.
- Revisit if: Chunked prefill (batched KV projection) is implemented.

### D-004: BOS is not a stop token for v2 models (2026-07-11)
- Context: The original `is_termination_token` treated both BOS and EOS as stop tokens. For Qwen3.5, BOS=0 is a valid content token.
- Options: A) keep BOS check for all models; B) only check BOS for v1 models.
- Decision: B. v2 models (Qwen3.5) also check `model_eos` and `chat_eos` from the v2 header.
- Because: Prevents premature termination on token 0 for Qwen3.5 while preserving Qwen3 behavior.
- Revisit if: Another v2 model uses a non-zero BOS as a stop token.
