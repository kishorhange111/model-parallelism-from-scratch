import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from mpar import pp, tp

P, B, L, D, H, V = 2, 2, 8, 16, 4, 12


def _init(tmp_path):
    return "file:///" + os.path.abspath(tmp_path / "rdzv").replace("\\", "/").lstrip("/")


def _spawn(fn, tmp_path, *args, n=P):
    mp.spawn(fn, args=(_init(tmp_path), n, *args), nprocs=n, join=True)


def _close(a, b):
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------- tensor parallelism
def _tp_worker(rank, init, n, sp):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=n)
    try:
        torch.manual_seed(0)
        full = tp.Block(D, H).double()
        x = torch.randn(B, L, D, dtype=torch.float64)
        dy = torch.randn(B, L, D, dtype=torch.float64)
        xr = x.clone().requires_grad_()
        full(xr).backward(dy)
        blk = tp.TPBlock(full, sp=sp).double()
        seq = slice(rank * L // n, (rank + 1) * L // n)
        xs = (x[:, seq] if sp else x).clone().requires_grad_()
        y = blk(xs)
        ref_y = full(x)
        _close(y, ref_y[:, seq] if sp else ref_y)
        y.backward(dy[:, seq] if sp else dy)
        _close(xs.grad, xr.grad[:, seq] if sp else xr.grad)
        d, hd = D, D // H
        heads = torch.arange(d).view(H, hd).chunk(n)[rank].flatten()
        rows = torch.cat([heads, heads + d, heads + 2 * d])
        _close(blk.qkv.weight.grad, full.qkv.weight.grad[rows])
        _close(blk.fc1.weight.grad, full.fc1.weight.grad.chunk(n)[rank])
        _close(blk.fc2.weight.grad, full.fc2.weight.grad.chunk(n, 1)[rank])
        _close(blk.proj.weight.grad, full.proj.weight.grad.chunk(n, 1)[rank])
        # replicated parameters: under SP each rank sees part of the sequence, so their
        # gradients are partial sums and need one all-reduce (as Megatron does)
        for mine, ref in ((blk.ln1.weight, full.ln1.weight), (blk.fc2.bias, full.fc2.bias)):
            g = mine.grad.clone()
            if sp:
                dist.all_reduce(g)
            _close(g, ref.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sp", [False, True])
def test_tensor_parallel_block_matches_single_device(tmp_path, sp):
    _spawn(_tp_worker, tmp_path, sp)


def _vocab_worker(rank, init, n):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=n)
    try:
        torch.manual_seed(0)
        emb = nn.Embedding(V, D).double()
        idx = torch.randint(0, V, (B, L))
        _close(tp.VocabParallelEmbedding(emb)(idx), emb(idx))
        logits = torch.randn(B * L, V, dtype=torch.float64) * 3
        tgt = torch.randint(0, V, (B * L,))
        full = logits.clone().requires_grad_()
        ref = F.cross_entropy(full, tgt)
        ref.backward()
        shard = logits.chunk(n, 1)[rank].clone().requires_grad_()
        loss = tp.vocab_parallel_cross_entropy(shard, tgt)
        _close(loss, ref.detach())
        loss.backward()
        _close(shard.grad, full.grad.chunk(n, 1)[rank])
    finally:
        dist.destroy_process_group()


def test_vocab_parallel_embedding_and_cross_entropy(tmp_path):
    _spawn(_vocab_worker, tmp_path)


# ----------------------------------------------------------------- pipeline parallelism
def _pp_worker(rank, init, n, sched, m):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=n)
    try:
        torch.manual_seed(0)
        blocks = nn.Sequential(*[tp.Block(D, H) for _ in range(n)]).double()
        head = nn.Linear(D, V).double()
        x = torch.randn(m * B, L, D, dtype=torch.float64)
        y = torch.randint(0, V, (m * B, L))
        loss_fn = lambda out, t: F.cross_entropy(head(out).flatten(0, 1), t.flatten())
        ref = loss_fn(blocks(x), y)
        ref.backward()
        ref_grads = [p.grad.clone() for p in blocks[rank].parameters()]
        blocks.zero_grad()
        mbs, tgts = list(x.chunk(m)), list(y.chunk(m))
        peak, loss = pp.run_pipeline(
            blocks[rank], pp.SCHEDULES[sched], mbs, tgts, loss_fn, (B, L, D), torch.float64
        )
        for p_, g in zip(blocks[rank].parameters(), ref_grads, strict=True):
            _close(p_.grad, g)
        if rank == n - 1:
            assert loss == pytest.approx(float(ref))
        assert peak == pp.peak_in_flight(pp.SCHEDULES[sched](n, m, rank))
        assert peak == (m if sched == "gpipe" else min(n - rank, m))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sched", ["gpipe", "1f1b"])
def test_pipeline_gradients_match_single_device(tmp_path, sched):
    _spawn(_pp_worker, tmp_path, sched, 6, n=3)


@pytest.mark.parametrize("p,m", [(4, 8), (4, 4), (8, 32), (2, 1)])
def test_bubble_fraction(p, m):
    for name, fn in pp.SCHEDULES.items():
        r = pp.simulate(fn, p, m)
        assert r["bubble"] == pytest.approx((p - 1) / (m + p - 1)), name


def test_schedules_are_complete_and_1f1b_bounds_memory():
    for p, m in ((4, 8), (3, 2)):
        for s in range(p):
            for fn in pp.SCHEDULES.values():
                ops = fn(p, m, s)
                assert sorted(ops) == sorted(
                    [("F", i) for i in range(m)] + [("B", i) for i in range(m)]
                )
            assert pp.peak_in_flight(pp.gpipe(p, m, s)) == m
            assert pp.peak_in_flight(pp.one_f_one_b(p, m, s)) == min(p - s, m)
