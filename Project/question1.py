from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
import sys
from typing import Callable
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from distributionhelper import Uniform


inpatient_arrival_rate = lambda hour: (3/8 + 27/8*(1-np.cos(2 * np.pi / 3 * (hour - 9)))) if 9 < hour < 15 else 3/8
outpatient_arrival_rate = lambda hour: 23/8 if 8 < hour < 16 else 0
emergency_patient_arrival_rate = 1

outpatient_show_prob = 0.84

    
def avg(func: Callable, a: float, b: float, steps: int = 1000) -> float:
    if b < a: raise ValueError(":a: cannot be greater than :b:")
    return sum(func(t) / (steps) for t in np.linspace(a, b, steps))
    
def C(c: int, rho: float) -> float:
    pi_c = (c*rho)**c / (math.factorial(c)*(1-rho))
    return pi_c / (pi_c + sum((c*rho)**j / math.factorial(j) for j in range(c))) 

def compute_num_scanners(scanner_cost: float, inpat_cost: float, outpat_cost: float, empat_cost: float, c_max: int, service_distr = Uniform(10/60,19/60), opt_rule: str = "avg") -> tuple[int, float, list[int]]:
    if not (opt_rule == "max" or opt_rule == "avg"): raise ValueError("Invalid opt_rule")
    E_B = service_distr.mean
    c_B = service_distr.std / E_B
    arrival_rate = lambda t :inpatient_arrival_rate(t) + outpatient_show_prob * outpatient_arrival_rate(t)
    min_val: float = float("inf")
    arg_min: int = 0
    
    yaxis = []
    xaxis = []
    
    invalid = []

    for c in trange(1, c_max + 1):
        rho_E = emergency_patient_arrival_rate * E_B / c
        rho = lambda t: arrival_rate(t) * E_B / c
        match opt_rule:
            case "max":
                rho_t = rho(10.5)
            case "avg":
                rho_t = avg(rho, 0, 24, 10_000)
        if rho_E + rho_t > 1:
            invalid.append(c)
            continue
        L_qE = (1 - c_B**2)* 0.5 * rho_E**2 / (1 - rho_E) + c_B**2 * rho_E / (1 - rho_E) * C(c, rho_E)
        W_qE = L_qE / emergency_patient_arrival_rate
        L_q = lambda t : (1 - c_B**2)* 0.5 * (rho_E + rho(t))**2 / (1 - rho_E - rho(t)) + c_B**2 * (rho_E + rho(t)) / (1 - rho_E - rho(t)) * C(c, rho_E + rho(t)) - L_qE
        W_q = lambda t : L_q(t) / arrival_rate(t)
        match opt_rule:
            case "max":
                W_qt = W_q(10.5)
            case "avg":
                W_qt = avg(W_q, 0, 24, 10_000)
        val = scanner_cost * c + (27*inpat_cost + 23*outpat_cost)*W_qt + 24*empat_cost * W_qE
        if val < min_val:
            min_val = val
            arg_min = c
        yaxis.append(val)
        xaxis.append(c)

    plt.plot(xaxis, yaxis)
    plt.show()

    return arg_min, float(min_val), invalid

if __name__ == "__main__":
    print(compute_num_scanners(10, 6, 0, 50, c_max=50))