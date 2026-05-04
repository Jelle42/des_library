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

        self.b_min: float = 1
        self.b: float = 10

        self.transaction_arrival_rate: float = 0.7
        self.block_arrival_rate: float = 1/12
        self.g: float = 10.69
        self.sigma_g: float = 0.5
        self.f: float = 4.56
        self.sigma_f: float = 0.3
        self.pi: float = 2

        self.T_exp = 300
    
    def insert_transaction(self, transaction: Transaction):
        keys = [tx.tip for tx in self.mempool]
        idx = bisect.bisect_left(keys, transaction.tip)
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

        self.expiry_event: Expire | None = None

class TransactionArrival(Event):
    def __init__(self, time: float, model: GasMarket) -> None:
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        demand = random.lognormvariate(m.g, m.sigma_g)
        max_fee = random.lognormvariate(m.f, m.sigma_f)
        tip = random.expovariate(m.pi)
        transaction = Transaction(m, demand, max_fee, tip)
        m.insert_transaction(transaction)

        expiry = Expire(self.time + m.T_exp, m, transaction)
        transaction.expiry_event = expiry
        sim.schedule(expiry)

        next_arrival = random.expovariate(m.transaction_arrival_rate)
        sim.schedule(TransactionArrival(next_arrival, m))

class Expire(Event):
    def __init__(self, time: float, model: GasMarket, transaction: Transaction) -> None:
        super().__init__(time)
        self.model = model
        self.transaction = transaction

    def execute(self, sim: Simulation) -> None:
        if self.cancelled:
            return
        self.model.mempool.remove(self.transaction)

class BlockArrival(Event):
    def __init__(self, time: float, model: GasMarket):
        super().__init__(time)
        self.model = model
    
    def execute(self, sim: Simulation) -> None:
        amount_gas_used: float = 0
        m = self.model

        for transaction in m.mempool:
            pass

        #update next base fee
        m.b = min(m.b_min, m.b*(1 + 1 / 8 * (amount_gas_used - m.gas_target) / m.gas_target))
