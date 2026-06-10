# KairoTransformers

KairoTransformers is the transformer-model package for the Kairo ML stack.

Phase 1 scope:

- configuration types,
- token/position metadata,
- attention shape validation,
- explicit dependency on tensor kernels in later integration.

Inference comes before training. Training requires correct tensor autodiff,
optimized matmul, normalization, scheduling, and checkpointing.
