# Teaching source map — KairoTransformers

Decoder/attention components built on bounded tensor and scheduling interfaces.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

## Reading order

1. [`README.md`](../README.md)
2. [`CMakeLists.txt`](../CMakeLists.txt)
3. [`src`](../src)
4. [`tests`](../tests)
5. [`research`](../research)

## Repository integration and API contracts

Tensor ops supply matmul/normalization; KairoONNX may import supported metadata; KairoGPU accelerates selected operations. Full training and model serving remain separate targets.

## Teaching standards

- Comments explain invariants, decision reasons, lifetime, prerequisites and cross-file effects; they do not narrate obvious statements.
- Follow upstream interfaces before implementation details; distinguish implemented, host-verified and planned features.
- Run available tests at a pinned commit and consult [STATUS.md](../STATUS.md) / [STATUS.yaml](../STATUS.yaml) before declaring completion.
- Reuse original contributions under [LICENSE](../LICENSE), subject to third-party code and asset terms.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (1)

1. [`CMakeLists.txt`](../CMakeLists.txt)

### Interfaces and modules (5)

1. [`src/KairoTransformerCheckpoint.cppm`](../src/KairoTransformerCheckpoint.cppm)
2. [`src/KairoTransformerRuntime.cppm`](../src/KairoTransformerRuntime.cppm)
3. [`src/KairoTransformers.cppm`](../src/KairoTransformers.cppm)
4. [`src/KairoTransformerTraining.cppm`](../src/KairoTransformerTraining.cppm)
5. [`src/KairoTransformerWeights.cppm`](../src/KairoTransformerWeights.cppm)

### Verification and tests (3)

1. [`tests/benchmark.cpp`](../tests/benchmark.cpp)
2. [`tests/smoke.cpp`](../tests/smoke.cpp)
3. [`tests/training.cpp`](../tests/training.cpp)

Snapshot totals: 9 first-party source/build/test files, 17 tracked files overall. Re-audit when files are renamed or added.
