from __future__ import annotations

import numpy as np
import math
from distributionhelper import *

def solve_traffic_eq(P: np.ndarray, gamma: np.ndarray):
    """
    Find lambda s.t. lambda_i = gamma_i + sum_j lambda_j P_ij
    """
    
    n = gamma.shape[0]
    if P.shape != (n, n): raise ValueError("P is not of desired shape")
    if (np.delete(P, -1, axis=0).sum(axis=1) != np.ones(n-1)).any(): raise ValueError("Rows of P do not sum to 1") 
    
    l = np.linalg.solve(np.identity(n) - P.T, gamma)
    return l

def C(c: int, rho: float) -> float:
    pi_c = (c*rho)**c / (math.factorial(c)*(1-rho))
    return pi_c / (pi_c + sum((c*rho)**j / math.factorial(j) for j in range(c))) 
                
def station_pmf(rho: float, c: int, pi_c: float, pi_0: float, kmax: int = 500):
    '''Returns the probability mass function of how many people have to wait'''
    arr = np.zeros(kmax+1)
    arr[0] = pi_0
    for k in range(1, kmax + 1):
        if k < c:
            arr[k] = (c * rho)**k / math.factorial(k) * pi_0
        elif k == c:
            arr[k] = pi_c
        else:
            arr[k] = rho**(k-c) * pi_c
    arr[0] = sum(arr[i] for i in range(c))
    arr = np.delete(arr, range(1,c))
    return arr


def find_num_chairs(arrival_rates: list[float]|np.ndarray, distributions: list[Distribution], servers: list[int], alpha: float, kmax: int = 500):
    if not (len(arrival_rates) == len(distributions) == len(servers)) :
        raise ValueError("*arrival_rates*, *distributions* and *servers* should have the same length")
    if not (0 <= alpha <= 1):
        raise ValueError("*alpha* should be a probability")
    
    n = len(arrival_rates)
    
    rho = [arrival_rates[i]*distributions[i].mean / 60 / servers[i] for i in range(n)]
    print(rho)
    
    pi_c: list[float] = [(1-rho[i])*C(servers[i], rho[i]) for i in range(n)]
    # print(pi_c)
    pi_0: list[float] = [math.factorial(servers[i]) / (servers[i]*rho[i])**servers[i] * pi_c[i] for i in range(n)]
    # print(pi)
    
    marginals: list[np.ndarray] = [station_pmf(rho[i], servers[i], pi_c[i], pi_0[i], kmax) for i in range(n)]
    
    total_pmf = marginals[0]
    for m in marginals[1:]:
        total_pmf = np.convolve(total_pmf, m)

    total_pmf = total_pmf / total_pmf.sum()

    cdf = np.cumsum(total_pmf)

    for k in range(kmax + 1):
        if 1 - cdf[k] < alpha:
            print("Chairs needed: ", k)
            break
    
    return cdf

if __name__ == "__main__":
    # service desk: 0, anesthesist: 1, CPM: 2, Additional tests: 3, home: 4
    P = np.array([
        [0, 0.75, 0.25, 0, 0],
        [0, 0, 0.0825, 0.19, 0.7275],
        [0, 0.751879699, 0, 0.090225564, 0.157894737],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    gamma = np.array([8.3, 0, 0, 0, 0])
    arrival_rates = [float(l) for l in solve_traffic_eq(P, gamma)]
    arrival_rates.pop()
    arrival_rates.insert(0, 1+2.625+2.415)
    find_num_chairs(arrival_rates, [Exponential(14.5), Gamma(1.01, 2.6931), Gamma(1.03, 18.7736), Gamma(1.02, 24.7861), Normal(10.27, 5.617)], [2, 1, 4 ,2, 1], 0.01, kmax=100)