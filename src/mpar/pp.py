"""Pipeline parallelism: GPipe and 1F1B schedules, a timing simulator and a real
multi-process executor.

The model is cut into p stages, one per rank, and a batch into m micro-batches. A stage
runs its op list in order: ``("F", i)`` receives activation i from the previous stage (or
reads micro-batch i) and sends its output on; ``("B", i)`` receives the output gradient,
back-propagates and sends the input gradient back.

- **GPipe** (Huang et al. 2019) runs all m forwards, then all m backwards. Each stage holds
  m micro-batches of activations at the peak.
- **1F1B** (PipeDream-Flush; Narayanan et al. 2021). Stage s runs p − s − 1 warm-up
  forwards, then alternates one forward with one backward. At most p − s micro-batches are
  ever in flight, so peak activation memory drops from O(m) to O(p).

Both have the same bubble, (p − 1)/(m + p − 1) of the time with F and B each costing
the same at every stage: 1F1B saves memory, not time. The simulator measures this and
the tests check it.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


def gpipe(p: int, m: int, s: int) -> list[tuple[str, int]]:
    return [("F", i) for i in range(m)] + [("B", i) for i in range(m)]


def one_f_one_b(p: int, m: int, s: int) -> list[tuple[str, int]]:
    warm = min(p - s - 1, m)
    ops = [("F", i) for i in range(warm)]
    f, b = warm, 0
    while f < m:  # steady state: one forward, one backward
        ops += [("F", f), ("B", b)]
        f, b = f + 1, b + 1
    ops += [("B", i) for i in range(b, m)]
    return ops


SCHEDULES = {"gpipe": gpipe, "1f1b": one_f_one_b}


def peak_in_flight(ops) -> int:
    live = peak = 0
    for kind, _ in ops:
        live += 1 if kind == "F" else -1
        peak = max(peak, live)
    return peak


def simulate(schedule, p: int, m: int, tf: float = 1.0, tb: float = 2.0) -> dict:
    """Event-driven timeline. F(s, i) needs F(s−1, i); B(s, i) needs B(s+1, i) (the last
    stage needs its own F). Each stage runs its ops in schedule order. Returns the makespan
    and bubble fraction."""
    ops = [schedule(p, m, s) for s in range(p)]
    done: dict = {}
    pc, free = [0] * p, [0.0] * p
    while any(pc[s] < len(ops[s]) for s in range(p)):
        progressed = False
        for s in range(p):
            if pc[s] == len(ops[s]):
                continue
            kind, i = ops[s][pc[s]]
            if kind == "F":
                dep = done.get(("F", s - 1, i), None) if s else 0.0
            else:
                dep = done.get(("B", s + 1, i)) if s < p - 1 else done.get(("F", s, i))
            if dep is None:
                continue
            start = max(free[s], dep)
            free[s] = start + (tf if kind == "F" else tb)
            done[(kind, s, i)] = free[s]
            pc[s] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("schedule deadlocks")
    span = max(free)
    return {"makespan": span, "bubble": 1 - m * (tf + tb) / span}


def run_pipeline(
    stage,
    schedule,
    microbatches,
    targets,
    loss_fn,
    act_shape,
    dtype=torch.float32,
    group=None,
    stats=None,
):
    """Execute ``schedule`` on this rank. ``microbatches`` / ``targets`` are only read on
    the first / last stage; ``act_shape`` is the activation shape passed between
    stages. Gradients accumulate in ``stage``'s parameters with loss / m.
    If ``stats`` is a dict, the seconds spent in forward and backward compute (not in
    waiting on ``recv``) are added to ``stats["busy"]``.
    Returns (peak micro-batches held, mean loss on the last stage or None)."""
    s, p = dist.get_rank(group), dist.get_world_size(group)
    clock = time.perf_counter
    busy = 0.0
    m = len(targets)
    held, peak, losses, sends = {}, 0, [], []
    for kind, i in schedule(p, m, s):
        if kind == "F":
            if s == 0:
                x = microbatches[i]
            else:
                x = torch.empty(act_shape, dtype=dtype)
                dist.recv(x, s - 1, group)
                x.requires_grad_()
            t0 = clock()
            y = stage(x)
            if s == p - 1:
                y = loss_fn(y, targets[i]) / m
                losses.append(float(y) * m)
            busy += clock() - t0
            if s < p - 1:
                sends.append(dist.isend(y.detach().contiguous(), s + 1, group))
            held[i] = (x, y)
            peak = max(peak, len(held))
        else:
            x, y = held.pop(i)
            if s == p - 1:
                t0 = clock()
                y.backward()
            else:
                g = torch.empty_like(y)
                dist.recv(g, s + 1, group)
                t0 = clock()
                y.backward(g)
            busy += clock() - t0
            if s > 0:
                sends.append(dist.isend(x.grad.contiguous(), s - 1, group))
    for w in sends:
        w.wait()
    if stats is not None:
        stats["busy"] = stats.get("busy", 0.0) + busy
    return peak, (sum(losses) / m if losses else None)
