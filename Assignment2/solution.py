from __future__ import annotations

import math
import random
import os
import sys
import bisect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class GasMarket:
    def __init__(self, gas_target: float, gas_limit: float, seed: int = 42):
        random.seed(seed)
        self.gas_target = gas_target
        self.gas_limit = gas_limit

        self.sim = Simulation()

        self.mempool: list[Transaction] = []

        self.b_min = 1
        self.b = 10

        self.arrival_rate: float = 0.7
        self.g: float = 10.69
        self.sigma_g: float = 0.5
        self.f: float = 4.56
        self.sigma_f: float = 0.3
        self.pi = 2
    
    def insert_transaction(self, transaction: Transaction):
        keys = [tx.tip for tx in self.mempool]
        idx = bisect.bisect_right(keys, transaction.tip)
        self.mempool.insert(idx, transaction)

class Transaction:
    def __init__(
            self,
            model: GasMarket,
            demand: float,
            max_fee: float,
            tip: float
        ) -> None:
        self.model = model
        self.demand = demand
        self.max_fee = max_fee
        self.tip = tip

class Arrival(Event):
    def __init__(self, time: float, model: GasMarket) -> None:
        super().__init__(time)
        self.model = model
        m = self.model
        demand = random.lognormvariate(m.g, m.sigma_g)
        max_fee = random.lognormvariate(m.f, m.sigma_f)
        tip = random.expovariate(m.pi)

        m.insert_transaction(Transaction(m, demand, max_fee, tip))

class Expire(Event):
    def __init__(self, time: float, model: GasMarket, transaction: Transaction) -> None:
        super().__init__(time)
        self.model = model