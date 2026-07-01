from __future__ import annotations

import numpy as np
from typing import Iterable
import math
import time
import random

from question4 import setup_sdp, compute_c

NUM_SLOTS = 32

def schedule_from_on_set(on_set: Iterable[int], num_slots: int = NUM_SLOTS) -> dict[int, int]:
    """Convert a set of 'open for outpatients' slot indices (1..num_slots)
    into the {slot: 0/1} dict that setup_sdp expects."""
    on_set = set(on_set)
    return {i: (1 if i in on_set else 0) for i in range(1, num_slots + 1)}
 
 
def warm_start_by_ci(c: list[float], k: int, num_slots: int = NUM_SLOTS) -> set[int]:
    """Heuristic start: keep the k slots with the LOWEST inpatient intensity c_i
    open for outpatients; block the highest-c_i slots (matches the 'block
    10-11 and 14-15' intuition from the project description)."""
    order = sorted(range(1, num_slots + 1), key=lambda i: c[i - 1])
    return set(order[:k])
 
 
def random_start(k: int, num_slots: int = NUM_SLOTS, rng: random.Random = random) -> set[int]:
    return set(rng.sample(range(1, num_slots + 1), k))
 
 
def swap_neighbor(on_set: set[int], num_slots: int = NUM_SLOTS, rng: random.Random = random) -> set[int]:
    """Neighbor move: swap one open slot for one closed slot (keeps |on_set| fixed)."""
    off_set = [i for i in range(1, num_slots + 1) if i not in on_set]
    if not on_set or not off_set:
        return set(on_set)
    i = rng.choice(list(on_set))
    j = rng.choice(off_set)
    new_on = set(on_set)
    new_on.remove(i)
    new_on.add(j)
    return new_on
 
 
# ---------------------------------------------------------------------------
# Objective (evaluate a schedule via the Q4 SDP), with memoization
# ---------------------------------------------------------------------------
 
def evaluate(on_set: set[int], c: list[float], params: dict, max_n: int,
             cache: dict[frozenset, float]) -> float:
    key = frozenset(on_set)
    if key in cache:
        return cache[key]
    schedule = schedule_from_on_set(on_set)
    sdp, known_vals = setup_sdp(c, params=params, outpatient_schedule=schedule, max_n=max_n)
    value = sdp.f((0, 0, 0), 0, known_vals)
    cache[key] = value
    return value
 
 
# ---------------------------------------------------------------------------
# Local search (simulated annealing over swap-neighborhoods)
# ---------------------------------------------------------------------------
 
def local_search(
    k: int,
    c: list[float],
    params: dict,
    max_n: int,
    num_iters: int = 150,
    num_restarts: int = 3,
    seed: int = 0,
    init_temp: float = 5.0,
    cooling: float = 0.97,
    verbose: bool = True,
) -> tuple[set[int], float, dict[frozenset, float]]:
    """
    Find S* = argmax f_0(0,0,0 | S) subject to S in {0,1}^32, sum(S) = k,
    via simulated-annealing local search with swap neighborhoods.
 
    Returns (best_on_set, best_value, cache) where cache maps every schedule
    evaluated (as a frozenset of open slots) to its SDP value, so you can
    inspect the search trace or reuse evaluations across calls.
    """
    rng = random.Random(seed)
    cache: dict[frozenset, float] = {}
    best_on, best_val = None, float("-inf")
 
    starts = []
    if 0 < k < NUM_SLOTS:
        starts.append(warm_start_by_ci(c, k))  # one informed start
    starts += [random_start(k, rng=rng) for _ in range(num_restarts)]
 
    for r, start in enumerate(starts):
        on_set = set(start)
        val = evaluate(on_set, c, params, max_n, cache)
        temp = init_temp
        for it in range(num_iters):
            cand = swap_neighbor(on_set, rng=rng)
            cand_val = evaluate(cand, c, params, max_n, cache)
            delta = cand_val - val
            if delta > 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
                on_set, val = cand, cand_val
            temp *= cooling
            if val > best_val:
                best_val, best_on = val, set(on_set)
                if verbose:
                    print(f"  [restart {r}] iter {it:3d}: new best {best_val:.3f}  (|S|={len(best_on)})")
        if verbose:
            print(f"restart {r} finished at value={val:.3f}  ({len(cache)} schedules evaluated so far)")
 
    return best_on, best_val, cache
 
if __name__ == "__main__":
    inpatient_arrival_rate = lambda t: 3 / 8 + 3 * (1 - np.cos(2 * np.pi / 3 * (t - 9))) if 9 <= t <= 15 else 3 / 8
    c = compute_c(inpatient_arrival_rate)
    params = {"r_O": 100, "r_I": 20, "w_O": 1.5, "w_I": 0, "pi_O": 10, "pi_I": 200}
 
    MAX_N_SEARCH = 10
    k = 16
 
    t0 = time.time()
    best_on, best_val, cache = local_search(
        k=k, c=c, params=params, max_n=MAX_N_SEARCH,
        num_iters=60, num_restarts=2, seed=0,
    )
    print(f"\nBest schedule for k={k}: {sorted(best_on)}")
    print(f"Best value: {best_val:.3f}  (evaluated {len(cache)} distinct schedules in {time.time()-t0:.1f}s)")
 
    # naive = evaluate(set(range(1, k + 1)), c, params, MAX_N_SEARCH, cache)
    # ci_heuristic = evaluate(warm_start_by_ci(c, k), c, params, MAX_N_SEARCH, cache)
    # print(f"Naive (first {k} slots) value:      {naive:.3f}")
    # print(f"c_i-heuristic (lowest-c_i slots):    {ci_heuristic:.3f}")
 