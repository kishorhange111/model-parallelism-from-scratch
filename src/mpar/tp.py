"""Megatron-LM tensor parallelism (Shoeybi et al. 2019) and sequence parallelism
(Korthikanti et al. 2022), written against ``torch.distributed``.

**The conjugate pair.** Everything is built from two autograd functions:

    f: forward identity,   backward all-reduce    (enter a tensor-parallel region)
    g: forward all-reduce, backward identity      (leave it)

An MLP ``Y = GeLU(X A) B`` splits A by columns and B by rows. Then ``GeLU(X A_i)`` is
local, no communication is needed between the two matmuls, and a single all-reduce sums
the partial products ``GeLU(X A_i) B_i``. Attention splits by heads in the same way: QKV
is column-parallel and the output projection row-parallel. A Transformer layer therefore
needs **one all-reduce forward and one backward per sub-block**.

**Sequence parallelism.** LayerNorm and dropout in between are replicated under plain TP,
so their activations are stored P times. SP shards them along the sequence instead and
replaces f/g by an all-gather / reduce-scatter pair (each the other's adjoint). The
communication volume is the same, since all-reduce = reduce-scatter + all-gather, and
those activations shrink by P.

**Vocab-parallel cross-entropy.** The logits stay sharded over the vocabulary. The global
max, the sum of exponentials and the target logit each need one all-reduce of a scalar per
token, never the full ``(tokens, V)`` matrix. The backward pass is local:
``softmax − onehot`` on each shard.

Collectives use only all-reduce and all-gather so they run on ``gloo`` (CPU). On GPUs
reduce-scatter would be NCCL's native op.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _ws(group=None):
    return dist.get_world_size(group)


def _rank(group=None):
    return dist.get_rank(group)


# -------------------------------------------------------------------- conjugate operators
class _F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        dist.all_reduce(g, group=ctx.group)
        return g, None


class _G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        x = x.clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None


def _all_gather(x, dim, group):
    parts = [torch.empty_like(x) for _ in range(_ws(group))]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim)


def _reduce_scatter(x, dim, group):
    x = x.clone()
    dist.all_reduce(x, group=group)  # gloo has no reduce_scatter; NCCL would use it
    return x.chunk(_ws(group), dim)[_rank(group)].contiguous()


class _AllGatherSeq(torch.autograd.Function):
    """SP entry: forward all-gather along the sequence, backward reduce-scatter."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_gather(x, 1, group)

    @staticmethod
    def backward(ctx, g):
        return _reduce_scatter(g, 1, ctx.group), None


class _ReduceScatterSeq(torch.autograd.Function):
    """SP exit: forward reduce-scatter along the sequence, backward all-gather."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _reduce_scatter(x, 1, group)

    @staticmethod
    def backward(ctx, g):
        return _all_gather(g, 1, ctx.group), None


def enter(x, group=None, sp=False):
    return _AllGatherSeq.apply(x, group) if sp else _F.apply(x, group)


def leave(x, group=None, sp=False):
    return _ReduceScatterSeq.apply(x, group) if sp else _G.apply(x, group)


# ------------------------------------------------------------------------------ layers
class ColumnParallelLinear(nn.Module):
    """Holds rows ``out/P`` of the full ``(out, in)`` weight; output stays sharded."""

    def __init__(self, full: nn.Linear, group=None, index=None):
        super().__init__()
        P, r = _ws(group), _rank(group)
        rows = index if index is not None else torch.arange(full.out_features).chunk(P)[r]
        self.weight = nn.Parameter(full.weight.detach()[rows].clone())
        self.bias = (
            nn.Parameter(full.bias.detach()[rows].clone()) if full.bias is not None else None
        )

    def forward(self, x):  # x already passed through ``enter``
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Holds columns ``in/P``; partial outputs are summed by ``leave``. The bias is added
    after the reduction, once."""

    def __init__(self, full: nn.Linear, group=None):
        super().__init__()
        P, r = _ws(group), _rank(group)
        self.weight = nn.Parameter(full.weight.detach().chunk(P, 1)[r].clone())
        self.bias = nn.Parameter(full.bias.detach().clone()) if full.bias is not None else None

    def forward(self, x_shard, group=None, sp=False):
        y = leave(F.linear(x_shard, self.weight), group, sp)
        return y + self.bias if self.bias is not None else y


# --------------------------------------------------------------------- reference model
class Block(nn.Module):
    """Pre-LN Transformer block used both as the single-device reference and as the
    source of the shards."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.ln1(x)
        b, l, d = h.shape
        q, k, v = self.qkv(h).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, l, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TPBlock(nn.Module):
    """The same block sharded over the group: heads split for attention, the hidden
    dimension split for the MLP. With ``sp=True`` the input and output are sequence shards
    ``(b, l/P, d)`` and the LayerNorms run on those shards."""

    def __init__(self, full: Block, group=None, sp=False):
        super().__init__()
        P, r = _ws(group), _rank(group)
        if full.h % P:
            raise ValueError("heads must be divisible by the TP degree")
        self.group, self.sp = group, sp
        d = full.qkv.in_features
        self.hd, self.h_local = d // full.h, full.h // P
        self.ln1, self.ln2 = (nn.LayerNorm(d) for _ in range(2))
        self.ln1.load_state_dict(full.ln1.state_dict())
        self.ln2.load_state_dict(full.ln2.state_dict())
        # QKV output is laid out [q | k | v], each (heads · hd); take this rank's heads of each
        heads = torch.arange(d).view(full.h, self.hd).chunk(P)[r].flatten()
        self.qkv = ColumnParallelLinear(
            full.qkv, group, torch.cat([heads, heads + d, heads + 2 * d])
        )
        self.proj = RowParallelLinear(full.proj, group)
        self.fc1 = ColumnParallelLinear(full.fc1, group)
        self.fc2 = RowParallelLinear(full.fc2, group)

    def forward(self, x):
        g, sp = self.group, self.sp
        h = enter(self.ln1(x), g, sp)
        b, l, _ = h.shape
        q, k, v = self.qkv(h).view(b, l, 3, self.h_local, self.hd).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        x = x + self.proj(y.reshape(b, l, -1), g, sp)
        h = enter(self.ln2(x), g, sp)
        return x + self.fc2(F.gelu(self.fc1(h)), g, sp)


# --------------------------------------------------------------- vocab-parallel pieces
class VocabParallelEmbedding(nn.Module):
    def __init__(self, full: nn.Embedding, group=None):
        super().__init__()
        P, r = _ws(group), _rank(group)
        V = full.num_embeddings
        if V % P:
            raise ValueError("vocab must be divisible by the TP degree")
        self.lo, self.hi = r * V // P, (r + 1) * V // P
        self.weight = nn.Parameter(full.weight.detach()[self.lo : self.hi].clone())
        self.group = group

    def forward(self, idx):
        local = (idx >= self.lo) & (idx < self.hi)
        e = F.embedding((idx - self.lo).clamp(0, self.hi - self.lo - 1), self.weight)
        return _G.apply(e * local[..., None], self.group)  # exactly one rank contributes


class _VocabParallelCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, lo, group):
        # logits (n, V/P) for vocab ids [lo, lo + V/P); target (n,) global ids
        m = logits.max(-1).values
        dist.all_reduce(m, dist.ReduceOp.MAX, group=group)
        e = torch.exp(logits - m[:, None])
        se = e.sum(-1)
        dist.all_reduce(se, group=group)
        local = (target >= lo) & (target < lo + logits.shape[1])
        t = torch.where(local, target - lo, torch.zeros_like(target))
        zt = logits.gather(1, t[:, None])[:, 0] * local
        dist.all_reduce(zt, group=group)
        loss = torch.log(se) + m - zt
        ctx.save_for_backward(e / se[:, None], t, local)
        return loss.mean()

    @staticmethod
    def backward(ctx, g):
        p, t, local = ctx.saved_tensors
        grad = p.clone()
        grad[torch.arange(len(t)), t] -= local.to(p.dtype)
        return grad * g / len(t), None, None, None


def vocab_parallel_cross_entropy(logits_shard, target, group=None):
    lo = _rank(group) * logits_shard.shape[1]
    return _VocabParallelCE.apply(logits_shard, target, lo, group)
