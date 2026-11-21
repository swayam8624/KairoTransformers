# KairoTransformers — linked source learning order

Begin with the Transformer configuration and Tensor interfaces, then parameter layout and runtime decoder computation, training/checkpoint serialization and smoke/reference tests. The design depends on Kairo's lower-level tensor math and does not imply foundation-model training scale.

## APIs and neighbors

Trace batch/tokens/masks -> embeddings -> attention/MLP/norm -> logits/loss, then parameter updates and checkpoint reload. Study causal masking, dimensions, numerical stability, deterministic seeds and the limits of current GPU acceleration.

## Read these source files (8)

1. [`src/KairoTransformers.cppm`](src/KairoTransformers.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
2. [`src/KairoTransformerWeights.cppm`](src/KairoTransformerWeights.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
3. [`src/KairoTransformerRuntime.cppm`](src/KairoTransformerRuntime.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
4. [`src/KairoTransformerTraining.cppm`](src/KairoTransformerTraining.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
5. [`src/KairoTransformerCheckpoint.cppm`](src/KairoTransformerCheckpoint.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
6. [`tests/smoke.cpp`](tests/smoke.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
7. [`tests/training.cpp`](tests/training.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
8. [`tests/benchmark.cpp`](tests/benchmark.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.

Read [README.md](README.md) for the toolchain and [KairoGameEngine](https://github.com/swayam8624/KairoGameEngine) for exact-head ecosystem acceptance. Original source is [MIT licensed](LICENSE) for learning, modification and continued development, without relicensing dependencies/fixtures. This guide is a documented navigation plan, not comprehensive in-file comment coverage or proof of correctness.
