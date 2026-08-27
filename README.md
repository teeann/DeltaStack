<h1 align="center">Push, Pop, Parallelize: Stack-Augmented Linear Attention via the Delta Rule</h1>

<p align="center">
  <a href="https://teeann.github.io/">Anh T Nguyen</a>,
  <a href="https://scholar.google.com/citations?user=Ab64YYUAAAAJ&hl=en">Saleh Momeni</a>,
  <a href="https://ashutoshchaubey.com/">Ashutosh Chaubey</a>,
  <a href="https://changnxx.github.io/">Changnan Xiao</a>,
  <a href="https://www.cs.uic.edu/~liub/">Bing Liu</a>
</p>

<h3 align="center">ICML 2026</h3>

<p align="center">
  <!-- <a href="[URL_TO_PROJECT_PAGE]">[Project Page]</a> -->
  <!-- <a href="[URL_TO_ARXIV_PAPER]">[Paper]</a> -->
</p>

## Abstract

Linear attention architectures based on the Delta rule, such as DeltaNet and RWKV-7, combine Transformer-level performance with RNN-like efficiency and can provably solve regular language tasks. However, due to their fixed-size state, these models fundamentally struggle to capture the recursive, hierarchical structures that are intrinsic to natural languages. To bridge this gap, we introduce DeltaStack, a novel architecture that augments the associative memory of DeltaNet with a lightweight, differentiable stack. Unlike prior stack-augmented approaches that rely on sequential recurrence, DeltaStack formulates stack operations as linear delta-rule updates. This formulation enables a hardware-aware implementation that is fully parallelizable over sequence length, preserving the training efficiency of linear transformers. Theoretically, we prove that DeltaStack extends the expressivity of DeltaNet to model both regular and hierarchical languages. Empirically, our method outperforms DeltaNet and Stack-Attention on comprehensive formal language benchmarks. Furthermore, a 340M-parameter DeltaStack model trained on 15B tokens surpasses strong DeltaNet baselines in both perplexity and zero-shot downstream performance.

## Updates

- **2026-04-30**: Paper accepted to ICML 2026!

## Installation

Requires a CUDA GPU -- the interleave kernels are written in Triton. Reference
environment: Python 3.12, PyTorch 2.9, CUDA 12.6.

```bash
pip install -r requirements.txt
```

## Running experiments

Experiments are JSON configs in `experiment_configs/`, passed by name:

```bash
python run_experiment.py -c lab           # single GPU (`just lab`)
python run_experiment.py -c lab --multi   # 4 seeds across 4 GPUs
```

Metrics print to stdout; results and checkpoints are written to `results/` and
`checkpoints/`. Key config fields:

| Key | Meaning |
| --- | --- |
| `task` | Formal-language task (see `data_dir/`) |
| `model_name` | `deltastack`, `deltanet`, `gateddeltanet`, `deltaproduct`, `rwkv6`, `transformer_causal`, `stack_rnn`, `lstm`, ... |
| `stack_size` | Stack depth. **Must be a power of 2** -- the fused Triton kernel indexes it with `tl.arange` |
| `val_stack_size` | Eval-time stack depth (default 256), for length generalization beyond the trained depth |

Logging to [Weights & Biases](https://wandb.ai) is optional: set `WANDB_API_KEY`
in a `.env` file at the repository root.
