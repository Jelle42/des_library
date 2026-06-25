from __future__ import annotations

import numpy as np

def solve_traffic_eq(P: np.ndarray, gamma: np.ndarray):
    """
    Find lambda s.t. lambda_i = gamma_i + sum_j lambda_j P_ij
    """
    n = gamma.shape[0]
    assert P.shape == (n, n)
    l = np.linalg.solve(np.identity(n) - P.T, gamma)
    return l

if __name__ == "__main__":
    # service desk: 0, anesthesist: 1, CPM: 2, Additional tests: 3, home: 4
    P = np.array([
        [0, 0.75, 0.25, 0, 0],
        [0, 0, 0.0825, 0.295, 0],
        [0, 0.751879699248, 0.0902255639098, 0, 0.157894736892],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ])
    gamma = np.array([8.3, 0, 0, 0, 0])
    print(solve_traffic_eq(P, gamma))