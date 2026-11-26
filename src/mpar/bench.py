"""Measured pipeline bubble on real processes (gloo, one CPU thread per stage).

    python -m mpar.bench --out results

A p-stage MLP pipeline trains one global batch split into m micro-batches. Each rank
times the seconds it spends in forward/backward compute (``busy``) against the wall time
of the step. The **measured bubble** is ``1 − busy / wall``, averaged over ranks, which
is the idle fraction the closed form ``(p − 1)/(m + p − 1)`` predicts. The step time is
also compared against one process running the whole model on the whole batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from . import pp


def _stage(d, layers, seed):
    torch.manual_seed(seed)
    return nn.Sequential(*[m for _ in range(layers) for m in (nn.Linear(d, d), nn.GELU())])


def _worker(rank, p, init, cfg, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=p)
    try:
        d, B = cfg["d"], cfg["batch"]
        stage = _stage(d, cfg["layers"], seed=rank)
        g = torch.Generator().manual_seed(0)
        X = torch.randn(B, d, generator=g)
        Y = torch.randn(B, d, generator=g)
        res = []
        for m in cfg["ms"]:
            for name in ("gpipe", "1f1b"):
                xs, ys = list(X.chunk(m)), list(Y.chunk(m))
                walls, busys, peak = [], [], 0
                for rep in range(cfg["reps"] + 1):
                    stage.zero_grad(set_to_none=True)
                    st = {}
                    dist.barrier()
                    t0 = time.perf_counter()
                    peak, _ = pp.run_pipeline(
                        stage, pp.SCHEDULES[name], xs, ys, F.mse_loss, (B // m, d), stats=st
                    )
                    dist.barrier()
                    if rep:  # the first repetition warms up
                        walls.append(time.perf_counter() - t0)
                        busys.append(st["busy"])
                k = sorted(range(len(walls)), key=walls.__getitem__)[len(walls) // 2]
                res.append({"m": m, "schedule": name, "wall": walls[k], "busy": busys[k],
                            "peak": peak})  # fmt: skip
        torch.save(res, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def single_process(cfg, p, reps):
    torch.set_num_threads(1)
    model = nn.Sequential(*[_stage(cfg["d"], cfg["layers"], s) for s in range(p)])
    g = torch.Generator().manual_seed(0)
    X = torch.randn(cfg["batch"], cfg["d"], generator=g)
    Y = torch.randn(cfg["batch"], cfg["d"], generator=g)
    times = []
    for _ in range(reps + 1):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        F.mse_loss(model(X), Y).backward()
        times.append(time.perf_counter() - t0)
    return sorted(times[1:])[len(times[1:]) // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2, help="Linear+GELU layers per stage")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    cfg = {"d": a.d, "layers": a.layers, "batch": a.batch, "ms": a.ms, "reps": a.reps}
    t1 = single_process(cfg, a.p, a.reps)
    with tempfile.TemporaryDirectory() as tmp:
        init = "file:///" + os.path.abspath(os.path.join(tmp, "rdzv")).replace("\\", "/").lstrip(
            "/"
        )
        mp.spawn(_worker, args=(a.p, init, cfg, tmp), nprocs=a.p, join=True)
        per_rank = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(a.p)]
    rows = []
    for j, base in enumerate(per_rank[0]):
        wall = max(r[j]["wall"] for r in per_rank)
        bubble = sum(1 - r[j]["busy"] / r[j]["wall"] for r in per_rank) / a.p
        rows.append({
            "m": base["m"], "schedule": base["schedule"], "step_s": wall,
            "bubble_measured": bubble,
            "bubble_theory": (a.p - 1) / (base["m"] + a.p - 1),
            "speedup_vs_1": t1 / wall, "peak_stage0": base["peak"],
            "busy_s_mean": sum(r[j]["busy"] for r in per_rank) / a.p,
        })  # fmt: skip
        print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    L = [
        f"p = {a.p} stages × {a.layers} Linear({a.d}, {a.d}) + GELU, global batch {a.batch}, "
        f"float32, gloo, one CPU thread per stage. One process, whole model and batch: "
        f"{t1:.3f} s per step.",
        "",
        "| m | schedule | step (s) | measured bubble | (p − 1)/(m + p − 1) | speed-up vs 1 process "
        "| micro-batches held, stage 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['m']} | {r['schedule']} | {r['step_s']:.3f} | {r['bubble_measured']:.3f} "
            f"| {r['bubble_theory']:.3f} | {r['speedup_vs_1']:.2f}× | {r['peak_stage0']} |"
        )
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
