# Masked Diffusion LLM (C++)

A from-scratch C++ implementation of a masked diffusion language model — 
the goal is an architecture that produces output like an autoregressive 
model but generates all tokens in parallel, making inference potentially 
faster and more efficient at scale.

No external dependencies.

## Status

Early stage / active research.

## Build

```bash
g++ -O2 -o diffuse_model diffuse_model.cpp
```