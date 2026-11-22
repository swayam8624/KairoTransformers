# Teaching source map — KairoTransformers
**Deep source walkthrough:** [Tokens, attention, KV caching and training](TRANSFORMER_FROM_TOKENS_TO_TRAINING.md). [Advanced source lab: cached decoding, quantization and exact state](TRANSFORMER_MODEL_STATE_AND_QUANTIZATION_LAB.md).

Decoder/attention components built on bounded tensor and scheduling interfaces.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

## Detailed source tutorial: follow one token from ID to loss

1. [`src/KairoTransformers.cppm`](../src/KairoTransformers.cppm) — configuration/shape guards, masks, normalization, QKV projection, rotary encoding, attention, MLP and transformer block calculations.
2. [`src/KairoTransformerWeights.cppm`](../src/KairoTransformerWeights.cppm) — dense/quantized parameter formats and INT8/INT4 packing. Distinguish compressed data representation from a correct numerical kernel.
3. [`src/KairoTransformerRuntime.cppm`](../src/KairoTransformerRuntime.cppm) — tokenizer abstraction, byte tokenizer, embedding table, generation and KV-cache state.
4. [`src/KairoTransformerCheckpoint.cppm`](../src/KairoTransformerCheckpoint.cppm) — checkpoint save/load, dimensions, schema/version and actual serialized state. File persistence is not evidence of exact training resumption unless optimizer/RNG are present.
5. [`src/KairoTransformerTraining.cppm`](../src/KairoTransformerTraining.cppm) — autograd graph, loss, optimizer, and bounded low-rank fine-tuning path.
6. [`tests/smoke.cpp`](../tests/smoke.cpp), [`tests/training.cpp`](../tests/training.cpp), [`tests/benchmark.cpp`](../tests/benchmark.cpp) — numerics, shape failures, training updates and timing evidence. Test *source* is not a test *run*.

**Cross-file call trace:** token IDs → embedding lookup → attention projection and RoPE → causal attention with optional KV history → residual/normalization/MLP layers → logits → optional training loss and backpropagation → updated weights and checkpoint. The runtime and training modules consume data layouts established by the model/weight modules; KairoMath handles owning tensor values, and unsupported backends do not become functional merely by being described in a configuration enum.

**Exercises:** derive attention score shapes for one head and a multi-head batch; trace the `past_length` index through a KV cache append; compare an unquantized weight path with dequantization and error tolerance; identify exactly which checkpoint fields restore optimizer/RNG state; verify whether generated-token sampling is deterministic with the supplied seed. Use fixed fixtures and run native tests before claiming numerical parity.

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
