from __future__ import annotations

import numpy as np

class DPFiniteHorizon:
    def __init__(
            self,
            state_space: np.ndarray,
            decision_space: list[np.ndarray],
            num_stages: int,
            transition_probs: list[list[np.ndarray]],
            immediate_costs: list[np.ndarray],
            objective: str = "max",
            traverse_asc: bool = True,
        ):
        """
        Defines a Dynamic Program with finite horizon.
        traverses stages from 1, ..., *num_stages* if *traverse_asc* = True, and from *num_stages*, *num_stages*-1, ..., 1 otherwise.
        """
        n = state_space.shape[0]
        if len(decision_space) != n:
            raise ValueError("decision_space must have one decision list per state")
        if len(transition_probs) != n:
            raise ValueError("transition_probs must have one action-probability list per state")
        if len(immediate_costs) != n:
            raise ValueError("immediate_costs must have one cost vector per state")

        self.state_space = state_space
        self.state_index = {state: idx for idx, state in enumerate(state_space)}
        self.decision_space = decision_space
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
        self.num_stages = num_stages
        self.objective = objective
        self.traverse_asc = traverse_asc

        for state_idx, decisions in enumerate(decision_space):
            if decisions.ndim != 1:
                raise ValueError(f"Decision space for state {state_idx} must be a 1D array")
            if len(transition_probs[state_idx]) != decisions.shape[0]:
                raise ValueError(
                    f"State {state_idx} has {decisions.shape[0]} decisions but {len(transition_probs[state_idx])} transition rows"
                )
            if immediate_costs[state_idx].shape != (decisions.shape[0],):
                raise ValueError(
                    f"Immediate costs for state {state_idx} must have shape {(decisions.shape[0],)}, got {immediate_costs[state_idx].shape}"
                )
            for action_idx, prob in enumerate(transition_probs[state_idx]):
                if prob.shape != (n,):
                    raise ValueError(
                        f"Transition probability vector for state {state_idx}, decision {action_idx} must have length {n}, got {prob.shape}"
                    )
                if (prob < 0).any():
                    raise ValueError(
                        f"Transition probs for state {state_idx}, decision {action_idx} contains negative values"
                    )
                if not np.isclose(prob.sum(), 1.0):
                    raise ValueError(
                        f"Transition probs for state {state_idx}, decision {action_idx} must sum to 1, got {prob.sum()}"
                    )
        if objective not in {"max", "min"}:
            raise ValueError("Invalid objective")

        self.optimal_decisions: dict[tuple[int, int], int] = {}

    def f(self, state: int, stage: int, known_values: np.ndarray) -> float:
        if state not in self.state_index:
            raise ValueError("Invalid state")
        state_idx = self.state_index[state]
        if stage == self.num_stages and self.traverse_asc or stage == 0 and not self.traverse_asc:
            return known_values[state_idx]

        objective_value = float("inf") if self.objective == "min" else float("-inf")
        optimal_decision = None
        next_stage = stage + 1 if self.traverse_asc else stage - 1

        for action_idx, decision in enumerate(self.decision_space[state_idx]):
            immediate_cost = self.immediate_costs[state_idx][action_idx]
            trans_prob = self.transition_probs[state_idx][action_idx]
            expected_future = sum(
                trans_prob[self.state_index[next_state]] * self.f(next_state, next_stage, known_values)
                for next_state in self.state_space
            )
            value = immediate_cost + expected_future

            if self.objective == "max":
                if value > objective_value:
                    objective_value = value
                    optimal_decision = decision
            else:
                if value < objective_value:
                    objective_value = value
                    optimal_decision = decision

        if optimal_decision is None:
            raise Exception("No valid decision found")

        self.optimal_decisions[(stage, state)] = optimal_decision
        return objective_value

    def solve(self, known_values: np.ndarray) -> np.ndarray:
        if known_values.shape != self.state_space.shape:
            raise ValueError(
                f"known_values is not of desired shape, expected {self.state_space.shape}, got {known_values.shape}"
            )
        desired_stage = 1 if self.traverse_asc else self.num_stages - 1
        return np.array([self.f(state, desired_stage, known_values) for state in self.state_space])
    
class DPInfiniteHorizon:
    def __init__(self,
            state_space: np.ndarray,
            decision_space: list[np.ndarray],
            transition_probs: list[list[np.ndarray]],
            immediate_costs: list[np.ndarray],
            objective: str = "max",
        ):
        """
        Defines a Dynamic Program with infinite horizon, also called a Markov Decision problem.
        """
        n = state_space.shape[0]
        if len(decision_space) != n:
            raise ValueError("decision_space must have one decision list per state")
        if len(transition_probs) != n:
            raise ValueError("transition_probs must have one action-probability list per state")
        if len(immediate_costs) != n:
            raise ValueError("immediate_costs must have one cost vector per state")

        self.state_space = state_space
        self.state_index = {state: idx for idx, state in enumerate(state_space)}
        self.decision_space = decision_space
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
        self.objective = objective

        for state_idx, decisions in enumerate(decision_space):
            if decisions.ndim != 1:
                raise ValueError(f"Decision space for state {state_idx} must be a 1D array")
            if len(transition_probs[state_idx]) != decisions.shape[0]:
                raise ValueError(
                    f"State {state_idx} has {decisions.shape[0]} decisions but {len(transition_probs[state_idx])} transition rows"
                )
            if immediate_costs[state_idx].shape != (decisions.shape[0],):
                raise ValueError(
                    f"Immediate costs for state {state_idx} must have shape {(decisions.shape[0],)}, got {immediate_costs[state_idx].shape}"
                )
            for action_idx, prob in enumerate(transition_probs[state_idx]):
                if prob.shape != (n,):
                    raise ValueError(
                        f"Transition probability vector for state {state_idx}, decision {action_idx} must have length {n}, got {prob.shape}"
                    )
                if (prob < 0).any():
                    raise ValueError(
                        f"Transition probs for state {state_idx}, decision {action_idx} contains negative values"
                    )
                if not np.isclose(prob.sum(), 1.0):
                    raise ValueError(
                        f"Transition probs for state {state_idx}, decision {action_idx} must sum to 1, got {prob.sum()}"
                    )
        if objective not in {"max", "min"}:
            raise ValueError("Invalid objective")
        
    def value_iter_discount(self, discount_factor: float = 0.95, epsilon: float = 1e-4) -> tuple[list[float], int]:
        obj = max if self.objective == "max" else min
        value_func: list[float] = [
            obj([
                self.immediate_costs[state_idx][decision]
                for decision in self.decision_space[state_idx]
                ]) for state_idx in range(len(self.state_space))
            ]
        iteration = 0
        optimal_decision: dict[int, int] = {}
        while True:
            iteration += 1
            prev_value_func = value_func
            for state_idx, state in enumerate(self.state_space):
                val = prev_value_func[state_idx]
                for decision in self.decision_space[state_idx]:
                    expr = self.immediate_costs[state_idx][decision] + discount_factor * sum(
                        self.transition_probs[state_idx][decision][j] * prev_value_func[j]
                        for j in range(len(self.state_space))
                        )
                    if expr < val and self.objective == "min" or expr > val and self.objective == "max":
                        value_func[state_idx] = expr
                        optimal_decision[state] = decision
            termination_check = max(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            if termination_check < epsilon * (1 - discount_factor) / (2*discount_factor): break
        return value_func, iteration
    
    def value_iter_average(self, epsilon: float = 1e-4):
        obj = max if self.objective == "max" else min
        value_func: list[float] = [
            obj([
                self.immediate_costs[state_idx][decision]
                for decision in self.decision_space[state_idx]
                ]) for state_idx in range(len(self.state_space))
            ]
        iteration = 0
        optimal_decision: dict[int, int] = {}
        while True:
            iteration += 1
            prev_value_func = value_func
            for state_idx, state in enumerate(self.state_space):
                val = prev_value_func[state_idx]
                for decision in self.decision_space[state_idx]:
                    expr = self.immediate_costs[state_idx][decision] +  sum(
                        self.transition_probs[state_idx][decision][j] * prev_value_func[j]
                        for j in range(len(self.state_space))
                        )
                    if expr < val and self.objective == "min" or expr > val and self.objective == "max":
                        value_func[state_idx] = expr
                        optimal_decision[state] = decision
            max_check = max(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            min_check = min(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            if 0 <= max_check - min_check <= epsilon * min_check: break
        return (min_check + max_check) / 2
    
if __name__ == "__main__":
    dp = DPInfiniteHorizon(
        state_space=np.array([0, 1, 2]),
        decision_space=[
            np.array([0, 1]),
            np.array([0, 1, 2]),
            np.array([0, 1])
        ],
        transition_probs=[
            [
                np.array([0.0, 0.0, 1.0]),
                np.array([0.9, 0.1, 0.0]),
            ],
            [
                np.array([0.5, 0.5, 0.0]),
                np.array([0.0, 0.8, 0.2]),
                np.array([0.1, 0.0, 0.9]),
            ],
            [
                np.array([0.2, 0.8, 0.0]),
                np.array([0.0, 0.3, 0.7]),
            ],
        ],
        immediate_costs=[
            np.array([2.0, 3.0]),
            np.array([1.0, 4.0, 2.0]),
            np.array([5.0, 1.0]),
        ],
    )
    values, it = dp.value_iter_discount(discount_factor=0.99, epsilon=1e-32)
    print(values)
    print(it)
    