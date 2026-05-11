from __future__ import annotations

import math
import random
import os
import sys
import bisect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class GasMarket:
    def __init__(self, transaction_arrival_rate: float, do_expire: bool = True, do_update_base_fee: bool = True, seed: int = 42):
        random.seed(seed)

        self.do_expire = do_expire
        self.do_update_base_fee = do_update_base_fee

        self.gas_target = 15 * 1e6
        self.gas_limit = 30 * 1e6

        self.num_blocks: int = 0
        self.num_transactions: int = 0

        self.sim = Simulation()

        self.mempool: list[Transaction] = []

        self.b_min: float = 1
        self.b: float = 10

        self.transaction_arrival_rate = transaction_arrival_rate
        self.block_arrival_rate: float = 1/12
        self.g: float = 10.69
        self.sigma_g: float = 0.5
        self.f: float = 4.56
        self.sigma_f: float = 0.3
        self.pi: float = 2

        self.T_exp: float = 300

        #statistics to keep track of
        self.confirmation_time = SampleStatistic()
        self.mempool_size = TimeWeightedStatistic()
        self.block_gas_utilisation = SampleStatistic()
        self.base_fee = TimeWeightedStatistic()
        self.num_expiries = Counter()
    
    def insert_transaction(self, transaction: Transaction):
        keys = [tx.tip for tx in self.mempool]
        idx = bisect.bisect_left(keys, transaction.tip)
        self.mempool.insert(idx, transaction)

    def run(self, stop: int | float, is_time: bool = True):
        self.sim.schedule(TransactionArrival(0.0, self))
        self.sim.schedule(BlockProduction(0.0, self))
        def stopping_condition1(sim: Simulation) -> bool:
            return sim.current_time > stop
        def stopping_condition2(sim: Simulation, model: GasMarket = self) -> bool:
            return model.num_blocks > stop
        if is_time:
            stop_con = stopping_condition1
        else:
            stop_con = stopping_condition2
        self.sim.run(stop_con)

    def report(self):
        t = self.sim.current_time
        print("Ethereum Gas Market Model")
        print(f"Horizon time: {t:.4f}")
        print(f"Avg. confirmation time: {self.confirmation_time.mean():.4f}")
        print(f"Avg. mempool size: {self.mempool_size.mean(t):.4f}")
        print(f"Avg. block-gas utilisation {self.block_gas_utilisation.mean():.4f}")
        print(f"Avg. base fee {self.base_fee.mean(t):.4f}")
        print(f"Avg. expiry rate {self.num_expiries.rate(t):.4f}")
        print(f"Number of blocks: {self.num_blocks}")
        print(f"Number of transactions: {self.num_transactions}")

class Transaction:
    def __init__(
            self,
            model: GasMarket,
            arrival_time: float,
            demand: float,
            max_fee: float,
            tip: float
        ) -> None:
        self.model = model
        self.arrival_time = arrival_time
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
        m.mempool_size.update(self.time, len(m.mempool))

        demand = random.lognormvariate(m.g, m.sigma_g)
        max_fee = random.lognormvariate(m.f, m.sigma_f)
        tip = random.expovariate(m.pi)
        transaction = Transaction(m, self.time, demand, max_fee, tip)
        m.insert_transaction(transaction)

        expiry = Expire(self.time + m.T_exp, m, transaction)
        transaction.expiry_event = expiry
        if m.do_expire: sim.schedule(expiry)

        next_arrival = random.expovariate(m.transaction_arrival_rate)
        sim.schedule(TransactionArrival(self.time + next_arrival, m))

        m.num_transactions += 1
        m.mempool_size.update(self.time, len(m.mempool))

class Expire(Event):
    def __init__(self, time: float, model: GasMarket, transaction: Transaction) -> None:
        super().__init__(time)
        self.model = model
        self.transaction = transaction

    def execute(self, sim: Simulation) -> None:
        if self.cancelled:
            return

        m = self.model
        m.mempool_size.update(self.time, len(m.mempool))
        self.model.mempool.remove(self.transaction)
        m.mempool_size.update(self.time, len(m.mempool))

        m.num_expiries.increment()

class BlockProduction(Event):
    def __init__(self, time: float, model: GasMarket):
        super().__init__(time)
        self.model = model
    
    def execute(self, sim: Simulation) -> None:
        amount_gas_used: float = 0
        m = self.model


        for i, transaction in enumerate(m.mempool):
            if transaction.demand + amount_gas_used > m.gas_limit: continue
            if transaction.expiry_event: transaction.expiry_event.cancel()
            amount_gas_used += transaction.demand
            m.mempool.pop(i)
            m.confirmation_time.record(self.time - transaction.arrival_time)

        m.block_gas_utilisation.record(amount_gas_used / m.gas_limit)

        if m.do_update_base_fee: sim.schedule(UpdateBaseFee(self.time, m, amount_gas_used))

        next_block_time = random.expovariate(m.block_arrival_rate)
        m.num_blocks += 1
        sim.schedule(BlockProduction(self.time + next_block_time, m))

class UpdateBaseFee(Event):
    def __init__(self, time: float, model: GasMarket, amount_gas_used):
        super().__init__(time)
        self.model = model
        self.amount_gas_used = amount_gas_used

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.base_fee.update(self.time, m.b)
        #update next base fee
        m.b = max(m.b_min, m.b*(1 + 1 / 8 * (self.amount_gas_used - m.gas_target) / m.gas_target))
        m.base_fee.update(self.time, m.b)

if __name__ == "__main__":
    model = GasMarket(0.7)
    model.run(10_000, True)
    model.report()
