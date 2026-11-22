# KairoTransformers — one token through attention, caching and training

This teaching path follows the complete C++23 module family:
[`KairoTransformers.cppm`](../src/KairoTransformers.cppm),
[`KairoTransformerRuntime.cppm`](../src/KairoTransformerRuntime.cppm),
[`KairoTransformerTraining.cppm`](../src/KairoTransformerTraining.cppm),
[`KairoTransformerWeights.cppm`](../src/KairoTransformerWeights.cppm) and
[`KairoTransformerCheckpoint.cppm`](../src/KairoTransformerCheckpoint.cppm).

## 1. Validate the architecture before allocating a tensor

`TransformerConfig` defines vocabulary size, maximum context,
model width, head count and layer count. A valid head configuration
requires compatible widths; the attention head dimension is
`modelWidth / headCount`. `ParameterCountEstimate` is a formula-derived
size, not a measured GPU-resident allocation.

KairoMath `Tensor` owns rank/shape/stride storage, so a tensor
with shape [sequence,width] is distinct from one with a batch or
head dimension. Never infer tensor layout from parameter names.

## 2. Attention step by step

A decoder layer projects token hidden states to Query, Key and Value
matrices; attention scores approximately follow:

[
S = rac{QK^T}{sqrt{d_k}}, qquad
A = operatorname{softmax}(S + M), qquad
O = AV
]

Here `M` is the causal mask: token t cannot attend to later tokens.
A softmax implementation must use a stable subtraction before exponentials
to avoid overflow. Normalize and residual/FFN code then produce each
layer's hidden representation.

`CausalScaledDotProductAttention`, `FusedQKVProjection`,
`LayerNorm`, `GELU` and `FeedForward` are separate APIs.
This modularity matters when comparing with KairoMath's autograd
attention and optional GPU kernels.

## 3. Generation and KV cache

[`KairoTransformerRuntime.cppm`](../src/KairoTransformerRuntime.cppm)
defines `Tokenizer`, a bounded `ByteTokenizer`,
`DecoderModel` and `KVCache`.

A decoder repeatedly predicts next-token logits. `GenerateGreedy`
selects the most likely token; stochastic sampling uses a configured
temperature/seed and allowed token set. A KV cache saves attention
keys/values from earlier steps so each new token does not recompute
the entire history. Its shapes and positions must match the model,
and the current cached path restricts batch configuration.

**Exercise:** with identical weights and prompt, compare full-forward
logits to cached stepwise logits before asserting equivalence. Check
context overflow, cache clear, token out-of-vocabulary and reproducible
seed behavior.

## 4. Trainable decoder and low-rank adapters

[`KairoTransformerTraining.cppm`](../src/KairoTransformerTraining.cppm)
uses KairoMath `TensorAutograd` to construct differentiable decoder
layers, a softmax-cross-entropy next-token loss, gradient accumulation
and optimizer updates. Training state includes parameters, optimizer
moments and deterministic random state.

`LoRAProjection` preserves a frozen base matrix while learning a low-rank
update approximately (Delta W = (alpha/r)AB). The adapter rank r,
scaling alpha, and training parameter list determine the update. This
must not be confused with silently fine-tuning every model weight.

Gradient checkpointing trades extra backward recomputation for reduced
stored activation memory. It does not by itself guarantee faster training.

## 5. Checkpoint and continuation semantics

A proper training resume must restore the same parameter names/order,
shapes, optimizer moments, completed steps and RNG state. Loading only
weight matrices cannot reproduce an interrupted Adam/AdamW run exactly.

Read [`tests/smoke.cpp`](../tests/smoke.cpp) and
[`tests/training.cpp`](../tests/training.cpp); then profile
[`tests/benchmark.cpp`](../tests/benchmark.cpp) with a fixed
model size, prompt length, generation mode and host.

## 6. Bounded claims

The implemented layers do not imply universal Hugging Face checkpoint
compatibility, all tokenizers, arbitrary sequence lengths or distributed
GPU pretraining. KairoONNX import and KairoGPU are optional integrations
with their own capability limits.

*Evidence note: code-reading guide; no new trained-model quality claim,
build result or native GPU benchmark is asserted here.*
