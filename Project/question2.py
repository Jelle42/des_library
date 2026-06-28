from __future__ import annotations

import math
from dpsolver import MDP
from scipy.stats import poisson


def find_dynamic_schedule(cost_per_scanner_per_daypart: float = 2.5, waiting_list_penalty: float = 10, arrivals_per_week: int = 5*23, num_scanners: int = 2, max_num_states: int = 100):
    states = list(range(max_num_states+1))
    decisions = {state: [16*i for i in range(10*num_scanners+1)] for state in states}
    
    def find_trans_probs(i: int, a: int) -> dict[int, float]:
        probs: dict[int, float] = {}
        
        for j in states[1:-1]:
            k = j - i + a
            probs[j] = float(poisson.pmf(k, arrivals_per_week)) if k >= 0 else 0.0
        probs[states[0]] = float(poisson.cdf(a - i, arrivals_per_week)) if a - i >= 0 else 0.0
        probs[states[-1]] =  1.0 - sum(probs[j] for j in states[:-1])
        
        return probs
    
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
    print(find_dynamic_schedule(cost_per_scanner_per_daypart=5, waiting_list_penalty=108, max_num_states=200))