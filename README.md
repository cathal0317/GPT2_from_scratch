# GPT2_from_scratch
Reimplemented GPT-2 from scratch in PyTorch

## Introduction

This project is a from-scratch reimplementation of GPT-2 in PyTorch.

The goal was not only to reproduce the architecture, but to deeply understand:
- how causal self-attention works
- how autoregressive language models are trained
- how pretrained weights map to custom implementations

This implementation closely follows the original GPT-2 design while keeping the code minimal and readable.

## Model Architecture

The model follows the standard GPT-2 architecture:

- Token + positional embeddings
- Stacked transformer blocks
  - LayerNorm
  - Causal self-attention
  - MLP (GELU activation)
- Residual connections throughout

Key implementation details:
- Multi-head attention implemented manually (no nn.MultiheadAttention)
- Causal masking using a lower triangular matrix
- Exact parameter shapes aligned with Hugging Face GPT-2

- ## Key Components

### Causal Self-Attention
Implements masked attention so tokens cannot attend to future positions.

### MLP Block
Two-layer feedforward network with GELU activation:
Linear → GELU → Linear

### Weight Loading
Custom model weights are mapped from Hugging Face GPT-2 checkpoints to ensure correctness.
