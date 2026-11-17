# TinkerLLM — Idea & Product Vision

> **Build, swap, and test language-model ideas in minutes.**

**Status:** Concept / proposed open-source project  
**Working name:** TinkerLLM (package name and availability to be verified)  
**Primary stack:** Python + pure PyTorch  
**Target users:** Architecture researchers, independent experimenters, students, and educators

## 1. The idea

TinkerLLM is a lightweight, extensible **LLM architecture research workbench**. Its purpose is to reduce the time between reading an architecture paper and running a faithful small-scale experiment.

A researcher should be able to:

- Build a conventional language model in a few lines by composing reusable components.
- Reproduce a published architecture without rewriting the entire training pipeline.
- Bring a **custom `torch.nn.Module`** and optionally describe it with a **dataclass configuration**.
- Change not only attention or FFN layers, but also residual pathways, state handling, routing, weight sharing, and **the model's execution flow**.
- Train small models and compare architectural hypotheses under controlled budgets.
- Learn architectural innovations through readable implementations, annotated mathematics, and runnable exercises.

**North star:** Make the *next* architecture easy to invent, not merely make existing architectures easy to select.

## 2. What TinkerLLM is — and is not

**It is:**

- A pure-PyTorch, research-first architecture toolkit.
- A collection of understandable reference implementations.
- A common experiment harness for controlled comparisons.
- An executable learning resource tracing the evolution of language-model architecture research.

**It is not (initially):**

- A production inference server or a replacement for vLLM.
- A custom CUDA/Triton kernel project.
- A distributed, billion-parameter training platform.
- A rigid configuration language that prevents unusual computation graphs.
- A guarantee that small-scale results reproduce large-scale paper claims.

## 3. Core design principles

1. **PyTorch first.** Every model and component is an inspectable `nn.Module`; minimal magic.
2. **Composition, not restriction.** Use built-ins where helpful, but allow arbitrary custom implementations and control flow.
3. **Config is not computation.** Dataclasses describe modules; PyTorch implements their behavior.
4. **Optional registration.** A registry can resolve a config to a module, but direct module injection must work.
5. **Reproducibility.** Record seeds, code revision, datasets, token budgets, hyperparameters, and measured results.
6. **Correctness before breadth.** Reference tests and mathematical fidelity matter more than a giant feature list.
7. **Research and education share the same core.** Tutorials use the actual research implementations, not a separate simplified engine.
8. **CPU and Apple Silicon friendly.** Prefer portable PyTorch; optimized backends can come later.

## 4. Researcher experience (proposed API, not implemented)

### Compose a model

```python
from tinkerllm import LanguageModel, Block, Repeat
from tinkerllm.nn import GQA, SwiGLU, RMSNorm

model = LanguageModel(
    vocab_size=32000,
    dim=512,
    body=Repeat(
        lambda: Block(
            mixer=GQA(dim=512, heads=8, kv_heads=2),
            ffn=SwiGLU(dim=512),
            norm=RMSNorm(dim=512),
        ),
        depth=8,
    ),
)
```

`Repeat` creates independent block instances; `Loop` (below) reuses the *same* block weights. A factory is used above to avoid accidental parameter sharing.

### Bring your own architecture

```python
from dataclasses import dataclass
import torch
from torch import nn
from tinkerllm import register

@dataclass
class CustomMixerConfig:
    dim: int = 512
    state_size: int = 64

@register(CustomMixerConfig)
class CustomMixer(nn.Module):
    def __init__(self, config: CustomMixerConfig):
        super().__init__()
        self.proj = nn.Linear(config.dim, config.dim)

    def forward(self, x, state=None):
        return self.proj(x), state  # Illustrative, not a novel mixer
```

The registry is a convenience. Researchers should also be able to supply a raw instance such as `mixer=CustomMixer(config)` without registering anything.

### Customize the entire execution flow

```python
from tinkerllm import Loop

model_body = Loop(
    block=shared_transformer_block,
    iterations=6,
)
```

A `Loop` is **not** just `Repeat`: it shares weights and may carry state across iterations. Future execution primitives can include conditional branches, recurrent scans, and learned halting.

### Compare hypotheses

```python
from tinkerllm.experiments import Experiment

study = Experiment(
    name="gqa-vs-mla",
    models={"gqa": gqa_model, "mla": mla_model},
    token_budget=20_000_000,
    seeds=[13, 37, 101],
)
study.run()
study.compare()
```

Fair comparisons should explicitly state whether models are **parameter-matched**, **compute-matched**, or **token-matched**; these are not interchangeable.

## 5. Coverage roadmap: chronological research map

The dates below are broad eras of research activity rather than an assertion that every technique originated within the stated window. This is a **coverage backlog**, not a promise that all items ship in v1.

| Era | Architectural ideas / example families | Primitives TinkerLLM should learn |
|---|---|---|
| **2017–2019** | Original Transformer, GPT, GPT-2, Transformer-XL, relative positions, sparse/local attention, early adaptive computation | MHA, causal masking, positional representations, residuals, recurrence |
| **2019–2021** | ReZero, RMSNorm, multi-query attention (MQA), linear attention, Performer, Linformer, Longformer, BigBird, Switch Transformer, GShard, RoPE, ALiBi | Alternate norms, positional methods, attention approximations, expert routing |
| **2021–2023** | PaLM-style parallel blocks, GLM / ChatGLM-style variants, LLaMA, SwiGLU, grouped-query attention (GQA), FlashAttention (algorithmic optimization), Mistral sliding-window attention, RetNet, RWKV, S4 | Parallel residual paths, grouped heads, local windows, recurrent/state-space abstractions |
| **2023–2024** | Mamba, Mamba-2, Griffin, Hyena, Mixtral MoE, DeepSeek-V2 MLA, DeltaNet, differential attention, hybrid attention/SSM families | Selective state updates, structured memory, latent KV projection, new mixers |
| **2024–2025** | DeepSeek-V3-style architectures, Gated DeltaNet, Qwen-family hybrid variations, Mixture-of-Depths, newer MoE routing and shared-expert patterns | Gated memory updates, hybrid layer schedules, conditional compute, sophisticated routing |
| **2025–2026 research frontier** | Recurrent-depth and looped Transformer variants, adaptive halting, novel residual pathways, attention–linear-attention hybrids, emerging GLM-family model designs | Arbitrary execution graphs, weight sharing, dynamic iteration, model-specific extension points |

**Important distinctions:**

- **GLM** can refer to more than one architecture or model family; document precisely which paper/version a reproduction targets.
- **FlashAttention** is principally an exact, IO-aware attention implementation, not a new attention semantics; it belongs in performance/implementation comparisons.
- Some looped/recurrent-depth ideas predate the recent frontier; they are placed later because they stress-test the framework's execution model.
- “DeepSeek-style” and “GLM-style” examples should specify their exact reproduced components and avoid claiming full paper fidelity when simplified.

### Coverage should be organized along two axes

**Historical examples** (to teach how ideas evolved) **and independent component families** (to let researchers recombine them):

- **Attention:** MHA, MQA, GQA, MLA, local/sliding window, linear attention, differential attention.
- **Recurrent/sequence mixers:** RWKV, RetNet, S4, Mamba, Mamba-2, DeltaNet, Gated DeltaNet, Hyena, hybrids.
- **FFNs and experts:** GELU FFN, SwiGLU/GeGLU, sparse MoE, routing, shared experts.
- **Position and normalization:** absolute/relative positions, RoPE, ALiBi, LayerNorm, RMSNorm, ReZero.
- **Execution and topology:** sequential stacks, parallel paths, shared-weight loops, conditional depth, recurrent state, adaptive halting.
- **Training/evaluation:** initialization variants, losses, optimizer configurations, matched-budget ablations, memory and throughput measurement.

## 6. TinkerLLM Learn

Every supported architecture can offer four complementary artifacts:

1. **Concept note:** what problem it was trying to solve and the mathematical mechanism.
2. **Readable source:** minimal reference PyTorch implementation, cross-linked to equations.
3. **Runnable lab:** a tiny model and controlled ablation on a small dataset.
4. **Checks:** shape tests, causality checks, numerical reference tests, and expected behavior.

Example learning path: MHA → MQA → GQA → MLA; softmax attention → linear attention → DeltaNet → Gated DeltaNet; Transformer → recurrent layers → looped Transformers.

Educational lessons should point out when an experiment is *illustrative*, rather than a statistically robust reproduction of published findings.

## 7. Experimental frontiers, including BioLLM-inspired adapters

TinkerLLM could also host unconventional **external-computation adapters**: modules that transform token or hidden-state signals using an external source of information or computation, then return results through a defined PyTorch-compatible interface.

Potential inputs might include recorded neural activity, neuromorphic signals, simulators, retrieval systems, or other experimental substrates. The biologically inspired case is speculative and **not an established method for improving language-model reasoning**.

A responsible first implementation would use **offline, prerecorded datasets** or a mocked adapter—not living neural hardware—and treat the component as an optional experiment, not a core requirement.

Research questions:

- Can an external signal improve a predefined benchmark over a matched random-signal or parameter-matched adapter control?
- Does any gain persist across seeds, datasets, and base model families?
- What latency, reproducibility, and generalization costs does the adapter introduce?

This extension is compatible with the architecture philosophy, but **should not distract from building reliable standard components first**.

## 8. Proposed repository structure

```text
tinkerllm/
├── pyproject.toml
├── README.md
├── src/tinkerllm/
│   ├── core/          # composition, configs, registry, execution
│   ├── nn/
│   │   ├── attention/
│   │   ├── mixers/
│   │   ├── ffn/
│   │   ├── moe/
│   │   ├── norm/
│   │   └── position/
│   ├── architectures/ # transparent, paper-specific examples
│   ├── training/      # portable tiny-model trainer
│   ├── evaluation/    # metrics, correctness, diagnostics
│   └── experiments/   # tracked ablations
├── examples/
├── learn/              # annotated labs and notes
└── tests/
```

## 9. Scaffolding strategy

**Start with a correct minimal Transformer, but do not jump from GPT-2 directly to looped Transformers.** Build chronologically *and* use selected architecture stress tests to validate extensibility.

### Milestone A — Runnable foundation

- Python package via `pyproject.toml`, `pip install -e .`, lint/test commands.
- Tiny causal Transformer and GPT-2-style reference architecture.
- Causal-mask tests, loss-decrease/overfit-one-batch smoke test, checkpoint round-trip.

### Milestone B — First architectural swaps

- MHA → MQA → GQA.
- LayerNorm → RMSNorm; positional embedding → RoPE/ALiBi; GELU FFN → SwiGLU.
- Direct `nn.Module` injection and dataclass-based config/registration.
- Small controlled comparison report.

### Milestone C — Modern blocks

- Sliding-window attention, standard MoE and routing, MLA.
- A clearly labeled DeepSeek-inspired composition demo.
- Tests for parameter sharing, routing counts, attention/cache correctness.

### Milestone D — Non-Transformer mixers

- A portable/reference linear-attention implementation.
- DeltaNet and Gated DeltaNet; then Mamba-family or another state-space implementation.
- Explicit prefill/step semantics and recurrent-state tests.
- Hybrid sequence mixer model without changing the core.

### Milestone E — New execution patterns

- `Loop`, shared-weight recurrent depth, optional adaptive halting.
- Demonstrate a looped Transformer and at least one unusual user-defined graph.
- New architecture can be added entirely in user code.

### Milestone F — Learn + research ecosystem

- Annotated architecture notebooks, paper references, experiment templates.
- Reproducible multi-seed ablations and comparison reports.
- Published tutorials and contribution guidelines for new architectures.

## 10. What counts as success?

**MVP acceptance criteria:**

- An unfamiliar user installs the package and trains a tiny LM on CPU/MPS.
- MHA, MQA, and GQA run behind one coherent, explicit interface.
- A researcher contributes a custom mixer using an `nn.Module` + optional dataclass, without changing core files.
- Tests catch causal leakage, accidental weight sharing, and incorrect state handling.
- One architecture comparison produces reproducible data and clearly states its budget controls.

**Longer-term proof of value:**

- Five external researchers can reproduce an unfamiliar architecture faster with TinkerLLM than with their usual baseline.
- At least one genuinely new computation pattern fits without a framework rewrite.
- Students can learn the evolution of architectures through executable reference implementations.
- Components remain readable and mathematically faithful as the catalog grows.

## 11. Risks and tradeoffs

| Risk | Mitigation |
|---|---|
| Overabstracting away research flexibility | Keep raw `nn.Module` support and ordinary Python control flow |
| Many superficial “paper reproductions” | Label exact scope, implement mathematical tests and paper-specific notes |
| Unfair architecture comparisons | Report tokens, parameters, FLOPs, latency, seeds, and budget-matching method |
| Recurrent models do not fit attention assumptions | Make execution/state interfaces extensible, not attention-specific |
| Breadth slows shipping | Release a small, tested core; keep a separate coverage backlog |
| Educational material diverges from code | Use the same implementations for tutorials and research experiments |

## 12. First release proposal

Release a **small, trustworthy PyTorch package**, not a 50-architecture catalog:

- GPT-2-style baseline and LLaMA-style block composition.
- MHA, MQA, GQA, RoPE, RMSNorm, SwiGLU.
- Dataclass configs, optional registry, and direct custom module injection.
- Portable trainer, tests, and one reproducible ablation example.
- A handful of TinkerLLM Learn lessons.

Then add MLA, MoE, DeltaNet, Gated DeltaNet, Mamba-family mixers, hybrid models, and looped/recurrent execution in follow-on releases.

---

**One-line pitch:** **TinkerLLM is the pure-PyTorch playground where you can learn yesterday's language-model architectures, reproduce today's, and invent tomorrow's.**
