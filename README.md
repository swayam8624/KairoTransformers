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

## Roadmap

1. Tensor-backed embeddings and rotary positions.
2. Decoder-only block composition and inference.
3. KV-cache incremental decoding.
4. Checkpoint loading.
5. Training after tensor autodiff and optimizer foundations are stable.
