# KairoTransformers

KairoTransformers is the transformer-model package for the Kairo ML stack. It
defines transformer configuration, shape planning, causal masking, and KV-cache
metadata before implementing full inference and training.

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
./build/KairoTransformersSmoke
```

## Roadmap

1. Tensor-backed embeddings.
2. LayerNorm and GELU kernels.
3. Scaled dot-product attention.
4. Multi-head attention and MLP blocks.
5. Decoder-only inference.
6. KV-cache incremental decoding.
7. Checkpoint loading.
8. Training after tensor autodiff and optimizer foundations are stable.
