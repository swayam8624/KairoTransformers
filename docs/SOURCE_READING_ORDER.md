# KairoTransformers — source-level learning sequence

This repository builds transformer attention, decoder blocks, model execution, weight handling, and selected training paths on [KairoMath tensors](https://github.com/swayam8624/KairoMath). It is **not** proof that a production-scale language-model inference stack, a tokenizer family, or every hardware backend is implemented.

## Learn by tracing one token

1. [README](../README.md), [CMakeLists.txt](../CMakeLists.txt) — supported features and test targets.
2. [src/KairoTransformers.cppm](../src/KairoTransformers.cppm) — `TransformerConfig`, dimensions/head-count invariants, causal masking, normalization, attention, MLP, and block formulas.
3. [src/KairoTransformerWeights.cppm](../src/KairoTransformerWeights.cppm) — stored dense/quantized weight representations, INT8/INT4 packing, and checkpoint/weight conversion.
4. [src/KairoTransformerRuntime.cppm](../src/KairoTransformerRuntime.cppm) — `Tokenizer` abstraction, `ByteTokenizer` implementation, embedding lookup, model forward/generation, input/output state, and KV cache handling where applicable.
5. [src/KairoTransformerCheckpoint.cppm](../src/KairoTransformerCheckpoint.cppm) — serialization and restore boundaries. Check weights, dimensions, versioning and optimizer state rather than trusting a filename extension.
6. [src/KairoTransformerTraining.cppm](../src/KairoTransformerTraining.cppm) — training graph, `TensorAutograd`, loss, optimizer updates, and low-rank adaptation as implemented.
7. [tests/smoke.cpp](../tests/smoke.cpp) — deterministic inference and shape/attention cases.
8. [tests/training.cpp](../tests/training.cpp) — gradients, updates, training state and learning checks.
9. [tests/benchmark.cpp](../tests/benchmark.cpp) — performance measurements; keep CPU/GPU, dataset and compiler context.

## Execution trace

Text bytes → `ByteTokenizer::Encode` tokens → token and position embeddings → Q/K/V projection → per-head causal attention with mask → output projection / residual → feed-forward block → logits → next-token selection → `Decode`. The public `Tokenizer` abstraction does not imply a BPE/SentencePiece implementation.

## Training trace

Inputs/targets → differentiable forward via `KairoMath::TensorAutograd` → loss and backward graph → optimizer updates on trainable weights/adapters → validation/checkpoint serialization. Quantized inference storage and trainable floating-point parameters have different numerical and mutability contracts.

## Concepts to verify while reading

- Shape of each tensor (sequence length, head count, head width, batch dimension).
- Mask direction and numerical behavior of softmax for fully masked/degenerate rows.
- Whether `KVCache` and quantized weights are actually used by the execution path being tested.
- Source of random seeds, reproducibility guarantees and memory ownership.
- Which tensors are copied, aliased, detached or persisted.

## Exercises

Trace one attention score end-to-end, test invalid head dimensions, compare quantized vs unquantized output, and restore a checkpoint before continuing training. Confirm tests and build settings locally.

Original code and documentation are proposed under [MIT](../LICENSE); checkpoint weights, datasets and imported model files retain their own rights.
