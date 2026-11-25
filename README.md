# model-parallelism-from-scratch

[![CI](https://github.com/kishorhange111/model-parallelism-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/model-parallelism-from-scratch/actions/workflows/ci.yml)

Megatron-style **tensor parallelism**, **sequence parallelism** and **pipeline parallelism**
(GPipe and 1F1B) written directly on `torch.distributed`. Each one is checked against
single-device training on a real multi-process group: outputs, input gradients and every
parameter-shard gradient agree to **1e-10** in float64.

## Tensor parallelism ([`tp.py`](src/mpar/tp.py))

Everything is built from one conjugate pair of autograd functions:

| | forward | backward |
|---|---|---|
| `f` (enter a TP region) | identity | all-reduce |
| `g` (leave it) | all-reduce | identity |

- **MLP.** `fc1` is column-parallel, so GeLU stays local. `fc2` is row-parallel, so one
  all-reduce sums the partial products. The bias is added once, after the reduction.
- **Attention.** QKV is split *by heads*: each rank takes its heads from each of the q, k
  and v blocks, and the output projection is row-parallel. A Transformer layer costs two
  all-reduces forward and two backward.
- **Sequence parallelism** (Korthikanti et al. 2022). LayerNorm and residual activations
  are sharded along the sequence instead of replicated. `f`/`g` become all-gather and
  reduce-scatter, each the other's adjoint. The communication volume is unchanged
  (all-reduce = reduce-scatter + all-gather), and those activations shrink P×. Replicated
  LayerNorm and bias gradients then become partial sums, so they need an all-reduce; the
  test shows this.
- **Vocab-parallel embedding and cross-entropy.** The logits stay sharded over V. The
  global max, the sum of exponentials and the target logit each need one all-reduce of a
  scalar per token, and the backward pass (`softmax − onehot`) is local.

## Pipeline parallelism ([`pp.py`](src/mpar/pp.py))

The schedules are op lists per stage: `("F", i)` and `("B", i)`. The executor runs them
with `isend` / `recv` of activations forward and of gradients backward.

| schedule | peak micro-batches held by stage s | bubble |
|---|---|---|
| GPipe | m | (p − 1)/(m + p − 1) |
| 1F1B | min(p − s, m) | (p − 1)/(m + p − 1) |

1F1B cuts activation memory from O(m) to O(p) **without changing the bubble**. An
event-driven simulator (F = 1, B = 2 time units) measures both against the closed form.
The executor's measured peak matches the schedule's.

## Tests

There are 10, on CPU, using `gloo` with 2–3 processes and float64. They check:
- a TP Transformer block, plain and with SP, matches the full block: output, dX and every
  shard of dW;
- vocab-parallel embedding and cross-entropy: loss and gradient match `F.cross_entropy`;
- a 3-stage pipeline with 6 micro-batches under GPipe and 1F1B gives gradients identical
  to full-batch training, the same loss, and the predicted peak memory;
- the bubble fraction equals (p − 1)/(m + p − 1) for both schedules;
- every schedule contains each F and B exactly once.

```
pip install -e ".[dev]" && pytest -q
```

Collectives use only all-reduce, all-gather and point-to-point ops so they run on
`gloo`. On GPUs, reduce-scatter would be NCCL's native op.
