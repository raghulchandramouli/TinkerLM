# TinkerLLM

> **Build, swap, and test language-model ideas in minutes.**

TinkerLLM is a lightweight, pure-PyTorch **LLM architecture research workbench** designed to shrink the gap between reading an architecture paper and running a faithful small-scale experiment.

## Repository Structure

```text
tinkerllm/
├── pyproject.toml
├── README.md
├── src/tinkerllm/
│   ├── core/            # Composition, configs, registry, execution primitives
│   ├── nn/
│   │   ├── attention/   # MHA, MQA, GQA, MLA, sliding-window, linear, differential
│   │   ├── mixers/      # Recurrent, SSM, DeltaNet, hybrid sequence mixers
│   │   ├── ffn/         # Standard & gated feedforward networks (GELU, SwiGLU, GeGLU)
│   │   ├── moe/         # Mixture-of-Experts layers, routers, shared experts
│   │   ├── norm/        # LayerNorm, RMSNorm, ReZero
│   │   └── position/    # Absolute/relative embeddings, RoPE, ALiBi
│   ├── architectures/   # Paper-specific reference compositions
│   ├── training/        # Portable tiny-model trainer
│   ├── evaluation/      # Metrics, correctness checks, diagnostics
│   └── experiments/     # Controlled budget-matched ablations
├── examples/            # Runnable scripts & demos
├── learn/               # Annotated concept notes & labs
└── tests/               # Shape, causality, state, and reference tests
```

## Development Setup

```bash
pip install -e ".[dev]"
```
