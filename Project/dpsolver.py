from __future__ import annotations

import numpy as np
from tqdm import tqdm

class DPFiniteHorizon:
    def __init__(self,
            state_space: np.ndarray,
            decision_space: np.ndarray,
            num_stages: int,
            transition_probs: list[np.ndarray],
            immediate_costs: np.ndarray,
            objective: str = "max",
            traverse_asc: bool = True,
        ):
        """
        Defines a Dynamic Program with finite horizon.
        traverses stages from 1, ..., *num_stages* if *traverse_asc* = True, and from *num_stages*, *num_stages*-1, ..., 1 otherwise.
        """
        n = state_space.shape[0]
        m = decision_space.shape[0]
        if len(transition_probs) != m:
            raise ValueError(f"transition_probs is not of desired shape, expected {m} matrices, got {len(transition_probs)}")
        for i,P in enumerate(transition_probs):
            if P.shape != (n,n):
                raise ValueError(f"Transition probs matrix {i+1} is not of desired shape, expected {(n,n)}, got {P.shape}")
            if (P < 0).any():
                raise ValueError(f"Transition probs matrix {i+1} is not stochastic; it contains elements < 0")
            for row in P:
                if row.sum() != 1:
                    raise ValueError(f"Transition probs matrix {i+1} is not stochastic; row {row} does not sum to 1")
        if immediate_costs.shape != (n,m):
            raise ValueError(f"immediate_costs is not of desired shape: is {immediate_costs.shape}, should be {(n,m)}")
        if not (objective == "max" or objective == "min"):
            raise ValueError("Invalide objective")
        self.state_space = state_space
        self.decision_space = decision_space
        self.num_stages = num_stages
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
        self.objective = objective
        self.traverse_asc = traverse_asc
        
        self.optimal_decisions: dict[tuple[int, int], int] = {} # maps (stage, state) to decision
        
    def f(self, state: int, stage: int, known_values: np.ndarray) -> float:
        if state not in self.state_space: raise ValueError("Invalid state")
        if stage == self.num_stages and self.traverse_asc or stage == 0 and not self.traverse_asc:
            return known_values[state]
        else:
            objective_value: float = float("inf") if self.objective == "min" else 0.0
            optimal_decision: int = self.decision_space[0]
            next_stage = stage + 1 if self.traverse_asc else stage - 1
            for idx, decision in enumerate(self.decision_space):
                value = self.immediate_costs[state, idx] + sum(self.transition_probs[idx][state,j]*self.f(j, next_stage, known_values) for j in self.state_space)
                if self.objective == "max":
                    if value > objective_value:
                        objective_value = value
                        optimal_decision = decision
                elif self.objective == "min":
                    if value < objective_value:
                        objective_value = value
                        optimal_decision = decision
            self.optimal_decisions[(stage, state)] = optimal_decision
            return objective_value
        
    def solve(self, known_values: np.ndarray) -> np.ndarray:
        if known_values.shape != self.state_space.shape: raise ValueError(f"known_values is not of desired shape, expected {self.state_space.shape}, got {known_values.shape}")
        desired_state = 1 if self.traverse_asc else self.num_stages - 1
        return np.array([self.f(i, desired_state, known_values) for i in self.state_space])
    
class DPInfiniteHorizon:
    def __init__(self, state_space: np.ndarray, decision_space: np.ndarray, transition_probs: list[np.ndarray], immediate_costs: np.ndarray, objective: str = "max", desired_result: str = "discounted"):
        n = state_space.shape[0]
        m = decision_space.shape[0]
        if len(transition_probs) != m:
            raise ValueError(f"transition_probs is not of desired shape, expected {m} matrices, got {len(transition_probs)}")
        for i,P in enumerate(transition_probs):
            if P.shape != (n,n):
                raise ValueError(f"Transition probs matrix {i+1} is not of desired shape, expected {(n,n)}, got {P.shape}")
            if (P < 0).any():
                raise ValueError(f"Transition probs matrix {i+1} is not stochastic; it contains elements < 0")
            for row in P:
                if row.sum() != 1:
                    raise ValueError(f"Transition probs matrix {i+1} is not stochastic; row {row} does not sum to 1")
        if immediate_costs.shape != (n,m):
            raise ValueError(f"immediate_costs is not of desired shape: is {immediate_costs.shape}, should be {(n,m)}")
        if not (objective == "max" or objective == "min"):
            raise ValueError("Invalid objective")
        if not (desired_result == "discounted" or desired_result == "avg"):
            raise ValueError("Invalid objective")
        self.state_space = state_space
        self.decision_space = decision_space
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
            
    
if __name__ == "__main__":
    dp = DPFiniteHorizon(
        state_space=np.array([0, 1, 2]),
        decision_space=np.array([0,1]),
        num_stages=5,
        transition_probs=[np.array([
            [0, 0, 1],
            [0.9, 0.1, 0],
            [1, 0, 0]
        ]) for _ in range(2)],
        immediate_costs=np.array([
            [2,3],
            [2,3],
            [2,3]
        ]),
    )
    print(dp.solve(np.array([1,1,5])))
    print([dp.optimal_decisions[1, state] for state in dp.state_space])
    