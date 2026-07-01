from __future__ import annotations

import math
import numpy as np
from dpsolver import MDP
from scipy.stats import poisson


def find_dynamic_schedule(cost_per_scanner_per_daypart: float = 1, waiting_list_penalty: float = 10, arrivals_per_week: int = 5*23, num_scanners: int = 2, max_num_states: int = 100):
    states = list(range(max_num_states+1))
    all_decisions = [16*i for i in range(10*num_scanners+1)]
    decisions = {state: all_decisions for state in states}

    outpatient_arrival_rate = arrivals_per_week
    inpatient_arrival_rate = 5 * 21
    emergency_arrival_rate = 5 * 8

    # k ~ Poisson(outpatient_arrival_rate), truncated far enough into the tail
    k_max = int(poisson.ppf(1 - 1e-10, outpatient_arrival_rate)) + 10
    k_pmf = poisson.pmf(np.arange(k_max + 1), outpatient_arrival_rate)

    # p_l = sum_{n+m=l} P(N=n) P(M=m), N ~ Poisson(inpatient), M ~ Poisson(emergency)
    n_max = int(poisson.ppf(1 - 1e-10, inpatient_arrival_rate)) + 10
    m_max = int(poisson.ppf(1 - 1e-10, emergency_arrival_rate)) + 10
    n_pmf = poisson.pmf(np.arange(n_max + 1), inpatient_arrival_rate)
    m_pmf = poisson.pmf(np.arange(m_max + 1), emergency_arrival_rate)
    l_pmf = np.convolve(n_pmf, m_pmf)

    # W_{t+1} = max(0, W_t + k + min(0, l - a)) = max(0, W_t + (k + min(l, a)) - a)
    s_pmf_cache: dict[int, np.ndarray] = {}
    for a in all_decisions:
        if a == 0:
            capped_l_pmf = np.array([1.0])  # min(l, 0) = 0 with probability 1
        else:
            head = l_pmf[:a] if a <= len(l_pmf) else np.concatenate([l_pmf, np.zeros(a - len(l_pmf))])
            tail_mass = max(0.0, 1.0 - head.sum())  # P(l >= a)
            capped_l_pmf = np.append(head, tail_mass)
        s_pmf_cache[a] = np.convolve(k_pmf, capped_l_pmf)

    def find_trans_probs(i: int, a: int) -> dict[int, float]:
        s_pmf = s_pmf_cache[a]
        next_states = np.clip(np.arange(len(s_pmf)) + (i - a), 0, states[-1])
        agg = np.zeros(len(states))
        np.add.at(agg, next_states, s_pmf)
        return {state: float(agg[idx]) for idx, state in enumerate(states)}

    trans_probs = {
        (state, decision): find_trans_probs(state, decision)
        for state in states for decision in decisions[state]
        }
    
    costs = {
        (state, decision): waiting_list_penalty*state + cost_per_scanner_per_daypart*decision / (16)
        for state in states for decision in decisions[state]
    }
    
    mdp = MDP(states, decisions, trans_probs, costs, objective="min")
    return mdp.policy_iter_average({state: 0 for state in states})

if __name__ == "__main__":
    policy, avg, it = find_dynamic_schedule(cost_per_scanner_per_daypart=40, waiting_list_penalty=108, max_num_states=200)
    last = -1
    for state, decision in policy.items():
        if decision == last: continue
        print(f"{state}: {decision}, {decision / 16}")
        last = decision
    print(f"avg: {avg}, iterations: {it}")