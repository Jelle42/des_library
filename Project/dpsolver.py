from __future__ import annotations

import numpy as np
from typing import Sequence, Mapping

# =================================================================================
# types for clarity:
# =================================================================================
StageType = int
StateType = int # describes states
StateSpaceType = Sequence[StateType] # collection of states
DecisionType = int # describes a decision
DecisionSpaceType = Mapping[StateType, Sequence[DecisionType]] # maps a state to all possible decisions in that state
TransitionProbabilitiesType = Mapping[tuple[StateType, DecisionType], Mapping[StateType, float]] # maps a tuple (state, decision) to a map state -> prob
SDPTransitionProbabilitiesType = Mapping[tuple[StateType, DecisionType, StageType], Mapping[StateType, float]] # maps a tuple (state, decision, stage) to a map state -> prob
CostFuncType = Mapping[tuple[StateType, DecisionType], float] # maps a tuple (state, decision) to an immediate cost/reward
# =================================================================================

# =================================================================================

class SDP:
    def __init__(
            self,
            state_space: StateSpaceType,
            decision_space: DecisionSpaceType,
            num_stages: StageType,
            transition_probs: SDPTransitionProbabilitiesType,
            immediate_costs: CostFuncType,
            objective: str = "max",
            traverse_asc: bool = True,
        ):
        """
        Defines a Dynamic Program with finite horizon.
        traverses stages from 1, ..., *num_stages* if *traverse_asc* = True, and from *num_stages*, *num_stages*-1, ..., 1 otherwise.
        """
        self.state_space = state_space
        self.decision_space = decision_space
        self.num_stages = num_stages
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
        self.objective = objective
        self.traverse_asc = traverse_asc
        
        # checks whether valid inputs were given
        # for (state, decision, stage), probs in transition_probs.items():
        #     if abs(sum(probs.values()) - 1) >= 1e-6:
        #         raise ValueError(f"State-decision-stage pair ({state}, {decision}, {stage}) has transition probabilities that do not sum to 1, instead they sum to {sum(probs.values())}.")
        #     for prob in probs.values():
        #         if prob < 0: raise ValueError(f"State-decision-stage pair ({state}, {decision}, {stage}) has negative transition probabilities.")

        if objective not in {"max", "min"}:
            raise ValueError("Invalid objective")

        self.optimal_decisions: dict[tuple[StateType, StageType], DecisionType] = {}
        self.computed_values: dict[tuple[StateType, StageType], float] = {}

    def f(self, state: StateType, stage: StageType, known_values: dict[StateType, float]) -> float:
        if (state, stage) in self.computed_values:
            return self.computed_values[(state, stage)]
        
        if state not in self.state_space:
            raise ValueError("Invalid state")
        if stage == self.num_stages and self.traverse_asc or stage == 0 and not self.traverse_asc:
            return known_values[state]

        objective_value = float("inf") if self.objective == "min" else float("-inf")
        optimal_decision = None
        next_stage = stage + 1 if self.traverse_asc else stage - 1

        for decision in self.decision_space[state]:
            immediate_cost = self.immediate_costs[(state, decision)]
            trans_prob = self.transition_probs[(state, decision, stage)]
            expected_future = sum(
                trans_prob[next_state] * self.f(next_state, next_stage, known_values) if next_state in trans_prob else 0.0
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

        self.optimal_decisions[(state, stage)] = optimal_decision
        self.computed_values[(state,stage)] = objective_value
        
        return objective_value

    def solve(self, known_values: dict[StateType, float]) -> tuple[dict[StateType, float], dict[tuple[StateType, StageType], DecisionType]]:
        if len(known_values) != len(self.state_space):
            raise ValueError(
                f"known_values is not of desired shape, expected {len(self.state_space)}, got {len(known_values)}"
            )
        desired_stage = 1 if self.traverse_asc else self.num_stages - 1
        return {state : self.f(state, desired_stage, known_values) for state in self.state_space}, self.optimal_decisions
    
class MDP:
    def __init__(self,
            state_space: StateSpaceType,
            decision_space: DecisionSpaceType,
            transition_probs: TransitionProbabilitiesType,
            immediate_costs: CostFuncType,
            objective: str = "max",
        ):
        """
        Defines a Dynamic Program with infinite horizon, also called a Markov Decision problem.
        """

        self.state_space = state_space
        self.state_index = {state: idx for idx, state in enumerate(state_space)}
        self.decision_space = decision_space
        self.transition_probs = transition_probs
        self.immediate_costs = immediate_costs
        self.objective = objective
        
        # checks whether valid inputs were given
        for (state, decision), probs in transition_probs.items():
            if len(probs.values()) != len(state_space):
                raise ValueError(f"Mismatch in number of states in transition probabilities of State-decision pair ({state}, {decision})")
            # if abs(sum(probs.values()) - 1) < 1e-6:
            #     raise ValueError(f"State-decision pair ({state}, {decision}) has transition probabilities that do not sum to 1.")
            # for next_state in state_space:
            #     if probs[state] < -1e-6: raise ValueError(f"State-decision pair ({state}, {decision}) has negative transition probability to state {next_state}.")

        if objective not in {"max", "min"}:
            raise ValueError("Invalid objective")
        
    def value_iter_discount(self, discount_factor: float = 0.95, epsilon: float = 1e-4) -> tuple[dict[StateType, float], dict[StateType, DecisionType], int]:
        obj = max if self.objective == "max" else min
        value_func: dict[StateType, float] = {
            state : obj([
                self.immediate_costs[(state, decision)]
                for decision in self.decision_space[state]
            ]) for state in self.state_space
            }
        iteration = 0
        optimal_decision: dict[StateType, DecisionType] = {}
        while True:
            iteration += 1
            prev_value_func = value_func.copy()
            for state in self.state_space:
                for decision in self.decision_space[state]:
                    expr = self.immediate_costs[(state, decision)] + discount_factor * sum(
                        self.transition_probs[(state, decision)][next_state] * prev_value_func[next_state]
                        for next_state in self.state_space
                        )
                    if expr < value_func[state] and self.objective == "min" or expr > value_func[state] and self.objective == "max":
                        value_func[state] = expr
                        optimal_decision[state] = decision
            termination_check = max(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            if termination_check < epsilon * (1 - discount_factor) / (2*discount_factor):
                break
        return value_func, optimal_decision, iteration
    
    def value_iter_average(self, epsilon: float = 1e-4) -> tuple[dict[StateType, float], dict[StateType, DecisionType], float, int]:
        obj = max if self.objective == "max" else min
        value_func: dict[StateType, float] = {
            state: obj([
                self.immediate_costs[(state, decision)]
                for decision in self.decision_space[state]
                ]) for state in self.state_space
        }
        iteration = 0
        optimal_decision: dict[StateType, DecisionType] = {}
        while True:
            iteration += 1
            prev_value_func = value_func.copy()
            for state in self.state_space:
                for decision in self.decision_space[state]:
                    expr = self.immediate_costs[(state, decision)] +  sum(
                        self.transition_probs[(state, decision)][next_state] * prev_value_func[next_state]
                        for next_state in self.state_space
                        )
                    if expr < value_func[state] and self.objective == "min" or expr > value_func[state] and self.objective == "max":
                        value_func[state] = expr
                        optimal_decision[state] = decision
            max_check = max(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            min_check = min(abs(value_func[i] - prev_value_func[i]) for i in self.state_space)
            if 0 <= max_check - min_check <= epsilon * min_check: break
        return value_func, optimal_decision, (min_check + max_check) / 2, iteration
    
    def policy_iter_discount(self, discount_factor: float, init_policy: dict[StateType, DecisionType]) -> tuple[dict[StateType, DecisionType], int]:
        r = [self.immediate_costs[(state, init_policy[state])] for state in self.state_space]
        P = np.array([[discount_factor * self.transition_probs[(state, init_policy[state])][next_state] for next_state in self.state_space]
            for state in self.state_space
        ])
        value_func: dict[StateType, float] = {self.state_space[i] : float(v) for i,v in enumerate(np.linalg.solve(np.identity(len(r)) - P, r))}
        policy = init_policy.copy()
        iteration = 0
        while True:
            iteration += 1
            prev_value_func = value_func.copy()
            prev_policy = policy.copy()
            
            for state in self.state_space:
                for decision in self.decision_space[state]:
                    expr = self.immediate_costs[(state, decision)] + discount_factor*sum(
                        self.transition_probs[(state, decision)][next_state]*prev_value_func[next_state]
                        for next_state in self.state_space
                    )
                    if expr > value_func[state] and self.objective == "max" or expr < value_func[state] and self.objective == "min":
                        value_func[state] = expr
                        policy[state] = decision
            
            if prev_policy == policy:
                break
        return policy, iteration
    
    def policy_iter_average(self, init_policy: dict[StateType, DecisionType]) -> tuple[dict[StateType, DecisionType], float, int]:
        n = len(self.state_space)
        s = 0 # arbitrarily chosen reference state, v_s = 0
        policy = init_policy.copy()
        iteration = 0
        
        while True:
            r = [self.immediate_costs[(state, policy[state])] for state in self.state_space]
            P = np.array([
                [self.transition_probs[(state, policy[state])][next_state] for next_state in self.state_space]
                for state in self.state_space
            ])
            A = np.identity(n) - P
            A = np.delete(A, s, axis=1)
            A = np.hstack([A, np.ones((n, 1))])
            solution = np.linalg.solve(A, r)
            v, g = solution[:-1], float(solution[-1])
            value_func: dict[StateType, float] = {self.state_space[i] : float(val) for i,val in enumerate(np.insert(v, s, 0.0))}
            iteration += 1
            
            prev_value_func = value_func.copy()
            prev_policy = policy.copy()
            
            for state in self.state_space:
                for decision in self.decision_space[state]:
                    expr = self.immediate_costs[(state, decision)] - g + sum(
                        self.transition_probs[(state, decision)][next_state]*prev_value_func[next_state]
                        for next_state in self.state_space
                    )
                    if expr > value_func[state] and self.objective == "max" or expr < value_func[state] and self.objective == "min":
                        value_func[state] = expr
                        policy[state] = decision
            
            if prev_policy == policy:
                break

        return policy, g, iteration
    
if __name__ == "__main__":
    num_stages = 10
    states = [0, 1, 2, 4]
    decisions = {
        0: [1, 2],
        1: [0, 4, 1],
        2: [2, 3],
        4: [0],
    }
    trans_probs = {
        (i, a): {j: 1 / len(states) for j in states} for i in states for a in decisions[i] 
    }
    trans_probs2 = {
        (i, a, n): {j: 1 / len(states) for j in states} for n in range(num_stages) for i in states for a in decisions[i]
    }
    immediate_costs = {
        (i, a): float(a) for i in states for a in decisions[i]
    }
    
    sdp = SDP(states, decisions, num_stages, trans_probs2, immediate_costs)
    print(sdp.f(states[0], 1, {state: 1/(state+1) for state in states}))