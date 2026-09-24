# KairoTransformers v1 Status

**Frozen v1 source completion: 95%.**

- Wave: `E`
- Frozen scope: `decoder-transformer-v1`
- Source gate: `complete`
- Exact-head execution: `pending_external_runner`
- Verification gate: `cmake-build-ctest-benchmark`
- Research track: `R7`
- Warning policy: `zero-kairo-owned-warnings`

The 95% score measures the bounded v1 implementation, integration contract,
tests/diagnostics surface and documentation. Native/platform execution evidence
is tracked separately and is never inferred from this score.

## Explicitly post-v1

- production tokenizer zoo
- safetensors mmap adapters
- large-model distributed inference

See `STATUS.yaml` for the machine-readable contract.
