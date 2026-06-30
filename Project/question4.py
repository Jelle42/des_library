from __future__ import annotations

import numpy as np
from typing import Callable, Literal
import random
import dpsolver as dp
from tqdm import tqdm
from scipy.integrate import quad



def integrate(f: Callable[[float], float], a: float, b: float) -> float:
    """return int_a^b f(x)dx"""
    return quad(f, a, b)[0]

def compute_c(func: Callable[[float], float], num_time_slots: int = 32) -> list[float]:
    return [integrate(func, 8 + i / 4, 8 + (i + 1) / 4) for i in range(num_time_slots)]

def setup_sdp(c: list[float], params: dict[str, float] = {"r_O": 100, "r_I": 20, "w_O": 1.5, "w_I": 0, "pi_O": 10, "pi_I":200}, outpatient_schedule: dict[dp.StageType, Literal[0,1]]|None = None, outpatient_prob: float|None = None, max_n: int = 100) -> tuple[dp.SDP, dict[dp.StateType, float]]:
    if outpatient_schedule is None and outpatient_prob is None:
        raise ValueError("Either outpatient_schedule or outpatient_prob should not be None")
    p_I = [1 - np.exp(-val) for val in c]
    p_E = 1 - np.exp(-0.25)
    p_s = 0.84
    states: list = [(i,j,k) for i in range(2) for j in range(max_n) for k in range(max_n)]
    decisions: dict = {}
    for (i, j, k) in states:
        if i > 0:
            decisions[(i,j,k)] = ["Emergency"]
        else:
            decisions[(i,j,k)] = []
            if j > 0:
                decisions[(i,j,k)].append("In")
            if k > 0:
                decisions[(i,j,k)].append("Out")
        if len(decisions[(i,j,k)]) == 0:
            decisions[(i,j,k)].append("Idle")
                
    transition_probs = {}
    for (n_E,n_I,n_O) in states:
        for a in decisions[(n_E,n_I,n_O)]:
            for stage in range(32):
                state_mapping: dict[tuple[int,int,int], float] = {}
                for (l,n,m) in states:
                    if outpatient_schedule is not None:
                        if outpatient_schedule[stage+1] == 1:
                            if a == "Emergency" or a == "Idle":
                                i = l - n_E
                                j = n - n_I
                                k = m - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and 0 <= k <= 1):
                                    continue
                            elif a == "In":
                                i = l - n_E
                                j = n - 1 - n_I
                                k = m - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and 0 <= k <= 1):
                                    continue
                            elif a == "Out":
                                i = l - n_E
                                j = n - n_I
                                k = m - 1 - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and 0 <= k <= 1):
                                    continue
                            state_mapping[(l,n,m)] = p_E**i * (1 - p_E)**(1-i) * p_I[stage]**j * (1 - p_I[stage])**(1-j) * p_s**k * (1 - p_s)**(1-k)
                        else:
                            if a == "Emergency" or a == "Idle":
                                i = l - n_E
                                j = n - n_I
                                k = m - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and k == 0):
                                    continue
                            elif a == "In":
                                i = l - n_E
                                j = n - 1 - n_I
                                k = m - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and k == 0):
                                    continue
                            elif a == "Out":
                                i = l - n_E
                                j = n - n_I
                                k = m - 1 - n_O
                                if not (0 <= i <= 1 and 0 <= j <= 1 and k == 0):
                                    continue
                            state_mapping[(l,n,m)] = p_E**i * (1 - p_E)**(1-i) * p_I[stage]**j * (1 - p_I[stage])**(1-j)
                
                transition_probs[((n_E, n_I, n_O), a, stage)] = state_mapping

    costs: dict[tuple[tuple[int,int,int], str], float] = {}
    for (n_E, n_I, n_O) in states:
        for a in decisions[(n_E, n_I, n_O)]:
            if a == "Emergency":
                costs[((n_E, n_I, n_O), a)] = - n_I * params["w_I"] - n_O * params["w_O"]
            elif a == "In":
                costs[((n_E, n_I, n_O), a)] = params["r_I"] - (n_I - 1)*params["w_I"] - n_O*params["w_O"]
            elif a == "Out":
                costs[((n_E, n_I, n_O), a)] = params["r_O"] - n_I*params["w_I"] - (n_O - 1)*params["w_O"]
            elif a == "Idle":
                costs[((n_E, n_I, n_O), a)] = 0.0
    
    known_states = {}
    for (n_E, n_I, n_O) in states:
        if n_E == 1:
            known_states[(n_E, n_I, n_O)] = -(params["w_I"] + params["pi_I"]) * n_I - (params["w_O"] + params["pi_O"])* n_O
        else:
            known_states[(n_E, n_I, n_O)] = max(
                params["r_I"] - (params["w_I"] + params["pi_I"]) * (n_I - 1) - (params["w_O"] + params["pi_O"])* n_O,
                params["r_O"] - (params["w_I"] + params["pi_I"]) * n_I - (params["w_O"] + params["pi_O"])* (n_O - 1)
            )
    
    return dp.SDP(states, decisions, 32, transition_probs, costs, "max", True), known_states

if __name__ == "__main__":
    inpatient_arrival_rate: Callable[[float], float] = lambda t: 3 / 8 + 3 * (1 - np.cos(2 * np.pi / 3 * (t - 9))) if 9 <= t <= 15 else 3 / 8
    c = compute_c(inpatient_arrival_rate) 
    print("c computed")
    sdp, known_vals = setup_sdp(c, outpatient_schedule={i: 0 if random.random() < 0.5 else 1 for i in range(33)}, max_n=32)
    print("sdp created")
    for state in tqdm(sdp.state_space):
        sdp.f(state, 0, known_vals)
    print("sdp solved")
    for (state, stage), decision in sdp.optimal_decisions.items():
        if stage != 0: continue
        if state[0] == 1: continue
        print(f"f_{stage}({state}) = {sdp.computed_values[(state, stage)]}, decision: {decision}")