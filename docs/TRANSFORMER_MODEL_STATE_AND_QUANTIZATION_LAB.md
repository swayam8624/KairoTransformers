# KairoTransformers — one token, two execution paths, three state lifetimes

A source-level laboratory for C++23 [model primitives](../src/KairoTransformers.cppm), [runtime and KV cache](../src/KairoTransformerRuntime.cppm), [packed weights](../src/KairoTransformerWeights.cppm), [autograd training](../src/KairoTransformerTraining.cppm), and [decoder weight archive](../src/KairoTransformerCheckpoint.cppm). Read these with the main [source learning sequence](SOURCE_READING_ORDER.md) and [the introductory transformer guide](TRANSFORMER_FROM_TOKENS_TO_TRAINING.md).

This tutorial distinguishes **implemented CPU tensor operations** from planned/high-performance kernels. A valid model configuration, quantized format, or module import is not evidence of a GPU-backed large-language-model server.

## 1. Why transformer configuration has strict shape rules

`TransformerConfig` contains vocabulary size `V`, context length `Tmax`, hidden width `D`, attention head count `H`, layer count `L` and feed-forward width `F`. `Valid()` rejects zero dimensions and requires `D % H == 0`; the logical per-head width is `Dh = D/H`.

For a sequence of `T` token IDs, token embedding gathers `T` rows from a `[V,D]` embedding table. Each decoder block accepts a `[T,D]` activation and returns the same shape. The LM head projects `[T,D]` into `[T,V]` logits. Loss functions can then compare each row's vocabulary probabilities with the one-hot next-token target.

The `AttentionShape` and `KVCacheDesc` helpers return *storage planning metadata*, not allocated buffers. The `ParameterCountEstimate` expression is an approximation based on a subset of model terms; it should not replace measuring the actual number/bytes of persisted tensors. Extreme configurations also need count-overflow tests.

## 2. Causal attention without a cache

For one head, the reference [`CausalScaledDotProductAttention`](../src/KairoTransformers.cppm) accepts query, key and value arrays of shape `[T,Dh]`. At output position `i`, it computes dot products only for past and present positions `j <= i`:

```text
score(i,j) = dot(query[i], key[j]) / sqrt(Dh)
weight(i,j) = exp(score(i,j) - max_past_score) / sum_past_exp
attended[i] = sum_{j<=i} weight(i,j) * value[j]
```

Subtracting the maximum prior to exponentiation improves numerical stability. Restricting `j <= i` enforces causality without explicitly generating a `[T,T]` future-mask matrix. The public `CausalMaskValue` helper separately expresses that policy as a large negative additive score for future positions.

[`MultiHeadCausalAttention`](../src/KairoTransformers.cppm) splits the `D` channels into `H` heads, executes that single-head reference loop on each, and repacks the results. `GroupedQueryCausalAttention` permits several Q heads to share a K/V head and requires `Hq % Hkv == 0`. The implementation remains a CPU reference; separate kernel fusion/dispatch tests are required before claiming optimized attention.

In the standard pre-norm decoder block, the actual call chain is `LayerNorm -> Q,K,V projections -> causal attention -> output projection -> residual add -> LayerNorm -> two-layer feed-forward -> residual add`. The `LayerNorm` function subtracts row means and normalizes variance; [Runtime::RMSNorm](../src/KairoTransformerRuntime.cppm) is different because it normalizes root-mean-square magnitude without subtracting the mean. `GELU` uses the tanh approximation rather than an exact normal CDF.

## 3. Incremental decoder KV state

[`KVCache`](../src/KairoTransformerRuntime.cppm) owns key and value tensors with logical dimensions `[layer, batch, context, KVheads, headWidth]` plus a length per layer. Its current constructor requires batch size one. Appending keys/values must happen at **exactly the stored length**, within context, with width `KVheads * headWidth`. This prevents overwriting an earlier position, skipping a token, or introducing a different shape midway through a generation trace.

`DecoderModel::Forward(tokens)` embeds and recomputes an entire prompt with stateless decoder blocks. `DecoderModel::Decode(token,position,cache)` instead embeds one token, computes rotary-position-modified Q and K, appends new K/V in every layer, attends over already-cached history, and returns the position's logits. `Generate` and `GenerateGreedy` first *prime* a fresh cache by replaying prompt tokens, then append new tokens in order.

The fundamental parity experiment is: build a tiny deterministic model and fixed prompt; compare the logits for the final token from `Forward(prompt)` against the final step of repeated `Decode(prompt[i],i,cache)`. Then extend the prompt by one token and repeat. Differences expose mistakes in causal masking, position offsets, RoPE, cache head sharing, normalization or tensor layouts. This comparison needs numerical tolerance and actual host execution; it was **not performed** during this documentation pass.

## 4. Greedy versus temperature/top-K/top-P sampling

`GreedyToken` returns the smallest token ID attaining maximum logit and never consumes random state. `SampleLogits` validates temperature and top-P, scales logits, sorts candidates by score with deterministic ID tie-breaking, applies optional top-K, exponentiates stabilized scores, then truncates to the cumulative top-P mass before sampling using `TrainingRandom`.

For repeatability, both logits and random seed/state must match. The same seed alone is insufficient when model parameters, prompt, quantization or preceding random draws differ. The returned token vector includes the original prompt and generated continuation.

The byte-level tokenizer implements a bounded `Tokenizer` interface; the presence of this abstraction does not imply compatibility with arbitrary BPE or sentencepiece vocabularies. Compare token IDs rather than assuming two tokenizers encode the same text identically.

## 5. Why INT8 and INT4 quantization cannot be treated as a free speedup

[`QuantizeInt8`](../src/KairoTransformerWeights.cppm) computes one symmetric scale for each output column of a `[inputWidth,outputWidth]` weight matrix, then rounds/clamps values to signed `[-127,127]`. Dequantization multiplies each stored signed integer by its column scale.

`QuantizeInt4` similarly scales to `[-7,7]`, shifts each value by eight, and stores two four-bit values in one byte. Even linear indices occupy low nibbles, odd indices high nibbles. During `QuantizedMatMul`, each packed nibble is unpacked and offset-corrected before accumulation, then the column scale is applied.

These are compressed **representations** and reference **Float32-accumulation kernels**. They need fidelity checks (max/relative error, logits and token prediction parity), storage measurement and a properly timed CPU/GPU comparison. Compression does not prove a measured throughput improvement; packing/dequantization and memory access can dominate small workloads.

## 6. Three kinds of checkpoints that must remain distinct

**Inference decoder weight archive.** `SaveDecoderCheckpoint` constructs a named list of embedding, final normalization, language-model head and every block's attention/MLP tensors, then saves through `BoundedTensorArchive`. `LoadDecoderCheckpoint` reads a configured number of layers under a per-tensor byte bound and returns `DecoderModelWeights`; the caller must supply compatible `TransformerConfig`. This file describes model **weights**, not the optimizer's step or RNG state.

**Autograd activation checkpointing.** `TrainableDecoder::SetGradientCheckpointing(true)` changes the computation graph so layer activations may be recomputed during backward to reduce retained forward state. It is a memory–compute tradeoff, not a saved file and not recoverability after a process crash.

**Trainable optimizer-state checkpoint.** `TrainableDecoder::Save/Load` delegates to `KairoMath::TensorTrainingCheckpoint` with parameters in stable order, the optimizer and `TrainingRandom`. The intent is restoration of parameters, optimizer state and RNG. Reproducible exact next-step parity still requires matching code, architecture, data order and numerical environment, verified by a real round-trip test.

These are different from MLLibrary's classic `KMLCP01` weights-only snapshot. Copying an inference archive into a training checkpoint loader is not a supported conversion strategy.

## 7. Backpropagation and gradient accumulation

[`TrainableDecoder`](../src/KairoTransformerTraining.cppm) stores autograd variables for embeddings, per-layer projections/MLP and the LM head. `Parameters()` returns borrowed pointers in a consistent order. `Forward(inputTokens)` performs differentiable embedding lookup, decoder layers and final projection; `Loss` converts targets to one-hot labels and builds softmax cross-entropy.

`TrainAccumulated` clears all parameter gradients once, invokes `Backward()` per supplied sequence, and calls `TensorOptimizer::Step` after accumulating all sequences, using their count as a divisor. It returns mean sequence loss, token accuracy on the provided sequences, and completed optimizer step count.

Gradient accumulation can simulate a larger training batch when loss weighting, optimizer scaling and stochastic behavior match, but do not assume bitwise parity with a single large-batch computation. The returned token accuracy is an in-sample diagnostic unless separate held-out sequences are evaluated.

## 8. Suggested test sequence and provenance

Begin with the checked-in [smoke tests](../tests/smoke.cpp), [training tests](../tests/training.cpp), and [benchmark tests](../tests/benchmark.cpp). Read each assertion before changing numerical precision, cache logic or weight persistence.

A meaningful extension suite should include a manually computed two-token causal attention example; full-sequence versus cached decode parity; grouped-query head-sharing checks; all-zero quantization, odd INT4 storage lengths and extreme scale values; deterministic greedy/sampled generation; optimizer checkpoint round-trip and *next step* equality; gradient-checkpointing-on/off loss/gradient parity; invalid archive sizes and tokenizer IDs.

Preserve commit SHA, host/compiler, seeded fixtures, raw numerical errors and test outputs. No native build, test execution, GPU measurement or numerical parity claim was established by writing this laboratory.
