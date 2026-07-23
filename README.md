# KairoTransformers

KairoTransformers is the transformer-model package for the Kairo ML stack. It
defines transformer configuration, Tensor-backed CPU inference primitives,
causal masking, and KV-cache metadata before implementing full inference and
training.

## Problem

Transformer support is often underestimated. Correct implementation needs:

- token and position handling,
- embeddings,
- causal and padding masks,
- LayerNorm/RMSNorm,
- rotary or learned positional encodings,
- multi-head attention,
- KV cache layout,
- MLP blocks,
- checkpoint loading,
- inference first, training later.

Training requires even more: autodiff, optimizer state, checkpointing, stable
normalization, fast matmul, scheduling, and memory planning.

## Solution

KairoTransformers starts with the pieces every later implementation needs:

- `TransformerConfig`: vocabulary, context, width, heads, layers, MLP size.
- `AttentionShape`: query/score element-count planning.
- `KVCacheDesc`: cache layout and memory sizing.
- `TokenBatch`: token-batch validation.
- `RopeConfig`: rotary-position metadata.
- `CausalMaskValue`: scalar causal-mask definition.
- `ParameterCountEstimate`: sizing and planning.
- Tensor-backed row-wise `LayerNorm` and GELU activation.
- Single-head causal scaled-dot-product attention over `[sequence, headWidth]`.
- `MultiHeadCausalAttention` for packed `[sequence, modelWidth]` Q/K/V
  tensors, using `TransformerConfig` head planning and per-head causal masking.
- `FeedForward` with explicit two-projection Tensor weights, configured
  ReLU/GELU/SiLU activation, and `AddResidual` composition.
- `DecoderBlockWeights` and `DecoderBlock`: a shape-checked pre-norm,
  decoder-only block with Q/K/V/output projections, causal multi-head
  attention, MLP, and both residual paths.
- token embedding lookup, RoPE, and affine RMSNorm primitives.
- `KVCache`: per-layer contiguous append and stable incremental attention.
- `DecoderModel`: multi-layer full-sequence logits, one-token cached decoding,
  and deterministic temperature/top-k/top-p generation.
- symmetric per-output-column INT8 dense weights with Float32 accumulation.
- `BoundedTensorArchive`: atomic indexed checkpoints that seek one tensor at a
  time under a caller-provided byte budget, plus layer-at-a-time streaming.

The runtime test verifies every cached token position against full-sequence
logits with RoPE enabled, repeats seeded generation exactly, bounds INT8 output
error, rejects over-budget archive reads, and proves layer streaming keeps only
the current layer map resident.

This keeps transformer model code separate from the tensor runtime while making
the required tensor shapes explicit.

## Where It Connects

- `KairoMath::Tensor`: stores embeddings, attention weights, activations, and
  KV caches.
- `KairoSIMD` and `KairoScheduler`: accelerate attention and MLP kernels on CPU.
- `KairoGPU`: eventually runs attention and matmul kernels on GPU.
- `KairoONNX`: imports external transformer weights and graphs.
- `MLLibrary`: exposes training/inference workflows.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
ctest --test-dir build --output-on-failure
./build/KairoTransformersSmoke
```

## Remaining Work

1. Grouped-query and multi-query attention cache layouts.
2. INT4 block quantization and memory-mapped safetensors metadata.
3. Autodiff transformer training, accumulation, checkpointing, and LoRA.
4. Tokenizer adapters and imported-checkpoint naming maps.
5. Tokens/second, peak RSS, and quantized numerical benchmark manifests.
