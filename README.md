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

### Measured on real processes ([`bench.py`](src/mpar/bench.py))

Four gloo processes, one CPU thread each, each holding two Linear(1024, 1024) + GELU
layers, train a global batch of 512 split into m micro-batches. Every rank times its
forward and backward compute (*busy*) against the step's wall time, and the measured
bubble is `1 − busy / wall`, averaged over ranks. One process running the whole model on
the whole batch takes 0.243 s ([results/bench.md](results/bench.md)).

| m | schedule | step (s) | measured bubble | (p − 1)/(m + p − 1) | busy per stage (s) | speed-up | micro-batches held, stage 0 |
|---|---|---|---|---|---|---|---|
| 1 | GPipe | 0.352 | 0.771 | 0.750 | 0.081 | 0.69× | 1 |
| 2 | GPipe | 0.213 | 0.621 | 0.600 | 0.081 | 1.14× | 2 |
| 4 | GPipe | **0.192** | 0.466 | 0.429 | 0.102 | **1.27×** | 4 |
| 8 | GPipe | 0.206 | 0.380 | 0.273 | 0.127 | 1.18× | 8 |
| 8 | 1F1B | 0.198 | 0.367 | 0.273 | 0.125 | 1.23× | **4** |
| 16 | GPipe | 0.252 | 0.264 | 0.158 | 0.186 | 0.96× | 16 |
| 16 | 1F1B | 0.251 | 0.266 | 0.158 | 0.184 | 0.97× | **4** |
| 32 | GPipe | 0.363 | 0.209 | 0.086 | 0.287 | 0.67× | 32 |
| 32 | 1F1B | 0.368 | 0.244 | 0.086 | 0.278 | 0.66× | **4** |

- **The bubble shrinks as predicted, but more micro-batches are not free.** The measured
  bubble tracks `(p − 1)/(m + p − 1)` within 0.04 up to m = 4. Beyond that it stays
  higher, by up to 0.16 at m = 32, because each micro-batch adds a send/recv round trip
  that the closed form does not count.
- **Per-stage compute grows 3.5× from m = 1 to m = 32 for the same FLOPs.** A 16-row
  matmul runs far less efficiently than a 512-row one. The best step time is therefore at
  m = 4 (1.27× over one process), not at the largest m. The ideal with four stages would
  be 4×. On GPUs the same trade-off sets the micro-batch size: it must stay large enough
  to keep the matmuls efficient.
- **1F1B has the same speed as GPipe and caps memory.** At every m the two schedules are
  within 0.04 s of each other (closer at large m), while stage 0 holds at most 4
  micro-batches under 1F1B against m under GPipe.

```
python -m mpar.bench --out results
```

## Tests

There are 11, on CPU, using `gloo` with 2–3 processes and float64. They check:
- a TP Transformer block, plain and with SP, matches the full block: output, dX and every
  shard of dW;
- vocab-parallel embedding and cross-entropy: loss and gradient match `F.cross_entropy`;
- a 3-stage pipeline with 6 micro-batches under GPipe and 1F1B gives gradients identical
  to full-batch training, the same loss, and the predicted peak memory;
- the bubble fraction equals (p − 1)/(m + p − 1) for both schedules;
- every schedule contains each F and B exactly once;
- the benchmark runs end to end and reports each schedule's micro-batch memory.

```
pip install -e ".[dev]" && pytest -q
```

Collectives use only all-reduce, all-gather and point-to-point ops so they run on
`gloo`. On GPUs, reduce-scatter would be NCCL's native op.
