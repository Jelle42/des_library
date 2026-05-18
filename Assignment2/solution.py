from __future__ import annotations

import math
import numpy as np
import random
import os
import sys
import bisect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class GasMarket:
    def __init__(
            self,
            transaction_arrival_rate: float,
            stopping_time: float,
            warmup_period: float = 0,
            block_arrival_rate: float | None = 1/12,
            do_expire: bool = True,
            do_update_base_fee: bool = True,
            block_capacity: int | None = None,
            seed: int = 42
        ):
        random.seed(seed)

        self.do_expire = do_expire
        self.do_update_base_fee = do_update_base_fee
        self.block_capacity = block_capacity
        self.stopping_time = stopping_time

        self.gas_target = 15 * 1e6
        self.gas_limit = 30 * 1e6

        self.num_blocks: int = 0
        self.num_transactions: int = 0

        self.sim = Simulation()

        self.mempool: list[Transaction] = []

        self.b_min: float = 1
        self.b: float = 10

        self.transaction_arrival_rate = transaction_arrival_rate
        self.block_arrival_rate = block_arrival_rate
        self.g: float = 10.69
        self.sigma_g: float = 0.5
        self.f: float = 4.56
        self.sigma_f: float = 0.3
        self.pi: float = 2

        self.T_exp: float = 300

        self.warmup_period = warmup_period

        self.num_batches: int = 50
        self.current_batch: int = 0 # runs from 0 to num_batches-1
        self.batch_times = np.linspace(self.warmup_period, self.stopping_time, self.num_batches)

        #statistics to keep track of
        self.confirmation_time = SampleStatistic()
        self.batch_confirmation_time = SampleStatistic()
        self.mempool_size = TimeWeightedStatistic()
        self.batch_mempool_size = SampleStatistic()
        self.block_gas_utilisation = SampleStatistic()
        self.batch_block_gas_utilisation = SampleStatistic()
        self.base_fee = TimeWeightedStatistic()
        self.batch_base_fee = SampleStatistic()
        self.num_expiries = Counter()
        self.batch_expiry_rate = SampleStatistic()

        #statistics for all sampled isntances
        self.gas_demands = SampleStatistic()
        self.batch_gas_demands = SampleStatistic()
        self.max_fees = SampleStatistic()
        self.batch_max_fees = SampleStatistic()
        self.tips = SampleStatistic()
        self.batch_tips = SampleStatistic()
        self.transaction_arrivals = SampleStatistic()
        self.batch_transaction_arrivals = SampleStatistic()
        self.block_arrivals = SampleStatistic()
        self.batch_block_arrivals = SampleStatistic()
    
    def insert_transaction(self, transaction: Transaction):
        keys = [tx.tip for tx in self.mempool]
        idx = bisect.bisect_right(keys, transaction.tip)
        self.mempool.insert(idx, transaction)

    def save_batch_statistics(self, time: float, reset: bool = True):
        # statistics to keep track of
        self.batch_confirmation_time.record(self.confirmation_time.mean())
        self.batch_mempool_size.record(self.mempool_size.mean(time))
        self.batch_block_gas_utilisation.record(self.block_gas_utilisation.mean())
        self.batch_base_fee.record(self.base_fee.mean(time))
        self.batch_expiry_rate.record(self.num_expiries.rate(time))

        # statistics for all samples instances
        self.batch_gas_demands.record(self.gas_demands.mean())
        self.batch_max_fees.record(self.max_fees.mean())
        self.batch_tips.record(self.tips.mean())
        self.batch_transaction_arrivals.record(self.transaction_arrivals.mean())
        self.batch_block_arrivals.record(self.block_arrivals.mean())

        if not reset: return
        self.confirmation_time.reset()
        self.mempool_size.reset()
        self.block_gas_utilisation.reset()
        self.base_fee.reset()
        self.num_expiries.reset()

        self.gas_demands.reset()
        self.max_fees.reset()
        self.tips.reset()
        self.transaction_arrivals.reset()
        self.block_arrivals.reset()
        
    def run(self):
        self.sim.schedule(TransactionArrival(0.0, self))
        self.sim.schedule(BlockProduction(0.0, self))
        self.sim.run(lambda sim: sim.current_time > self.stopping_time)

    def report(self):
        t = self.sim.current_time
        print("Ethereum Gas Market Model")
        print(f"Horizon time: {t:.4f}")
        print(f"Avg. confirmation time: {self.batch_confirmation_time.mean():.4f}")
        print(f"Avg. mempool size: {self.batch_mempool_size.mean():.4f}")
        print(f"Avg. block-gas utilisation {self.batch_block_gas_utilisation.mean():.4f}")
        print(f"Avg. base fee {self.batch_base_fee.mean():.4f}")
        print(f"Avg. expiry rate {self.batch_expiry_rate.mean():.4f}")
        print(f"Number of blocks: {self.num_blocks}")
        print(f"Number of transactions: {self.num_transactions}")
        print(f"Avg. gas demand: {self.batch_gas_demands.mean():.4f} vs True mean: {50_000}")
        print(f"Avg. max fee: {self.batch_max_fees.mean():.4f} vs True mean: {100}")
        print(f"Avg. tip: {self.batch_tips.mean():.4f} vs True mean: {1/self.pi:.4f}")
        print(f"Avg. transaction arrival rate {self.batch_transaction_arrivals.mean():.4f} vs True mean: {1/self.transaction_arrival_rate:.4f}")
        if self.block_arrival_rate is not None:
            print(f"Avg. Block arrival rate {self.batch_block_arrivals.mean():.4f} vs True mean: {1/self.block_arrival_rate:.4f}")
        else:
            print(f"Avg. Block arrival rate {self.batch_block_arrivals.mean():.4f} vs True mean: {12}")
        print(f"Current batch: {self.current_batch}")
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

        if self.time > m.batch_times[m.current_batch]:
            m.save_batch_statistics(self.time - m.batch_times[m.current_batch])
            m.current_batch += 1

        if self.time > m.warmup_period: m.mempool_size.update(self.time - m.batch_times[m.current_batch], len(m.mempool))

        demand = random.lognormvariate(m.g, m.sigma_g)
        max_fee = random.lognormvariate(m.f, m.sigma_f)
        tip = random.expovariate(m.pi)
        if self.time > m.warmup_period:
            m.gas_demands.record(demand)
            m.max_fees.record(max_fee)
            m.tips.record(tip)

        transaction = Transaction(m, self.time, demand, max_fee, tip)
        m.insert_transaction(transaction)

        expiry = Expire(self.time + m.T_exp, m, transaction)
        transaction.expiry_event = expiry
        if m.do_expire: sim.schedule(expiry)

        next_arrival = random.expovariate(m.transaction_arrival_rate)
        if self.time > m.warmup_period: m.transaction_arrivals.record(next_arrival)
        sim.schedule(TransactionArrival(self.time + next_arrival, m))

        m.num_transactions += 1
        if self.time > m.warmup_period: m.mempool_size.update(self.time - m.batch_times[m.current_batch], len(m.mempool))

class Expire(Event):
    def __init__(self, time: float, model: GasMarket, transaction: Transaction) -> None:
        super().__init__(time)
        self.model = model
        self.transaction = transaction

    def execute(self, sim: Simulation) -> None:
        if self.cancelled:
            return

        m = self.model
        if self.time > m.batch_times[m.current_batch]:
            m.save_batch_statistics(self.time - m.batch_times[m.current_batch])
            m.current_batch += 1

        if self.time > m.warmup_period: m.mempool_size.update(self.time - m.batch_times[m.current_batch], len(m.mempool))
        self.model.mempool.remove(self.transaction)
        if self.time > m.warmup_period: m.mempool_size.update(self.time - m.batch_times[m.current_batch], len(m.mempool))

        m.num_expiries.increment()

class BlockProduction(Event):
    def __init__(self, time: float, model: GasMarket):
        super().__init__(time)
        self.model = model
    
    def execute(self, sim: Simulation) -> None:
        amount_gas_used: float = 0
        m = self.model

        if self.time > m.batch_times[m.current_batch]:
            m.save_batch_statistics(self.time - m.batch_times[m.current_batch])
            m.current_batch += 1

        if m.block_capacity is not None:
            queue = m.mempool[-m.block_capacity:]
        else:
            queue = m.mempool

        for transaction in reversed(queue):
            if transaction.demand + amount_gas_used > m.gas_limit: continue
            if transaction.max_fee < m.b: continue
            if transaction.expiry_event: transaction.expiry_event.cancel()
            amount_gas_used += transaction.demand
            m.mempool.remove(transaction)
            if self.time > m.warmup_period: m.confirmation_time.record(self.time - transaction.arrival_time)

        if self.time > m.warmup_period: m.block_gas_utilisation.record(amount_gas_used / m.gas_limit)

        if m.do_update_base_fee: sim.schedule(UpdateBaseFee(self.time, m, amount_gas_used))

        if m.block_arrival_rate is not None:
            next_block_time = random.expovariate(m.block_arrival_rate)
        else:
            next_block_time = 12

        if self.time > m.warmup_period: m.block_arrivals.record(next_block_time)

        m.num_blocks += 1
        sim.schedule(BlockProduction(self.time + next_block_time, m))

class UpdateBaseFee(Event):
    def __init__(self, time: float, model: GasMarket, amount_gas_used):
        super().__init__(time)
        self.model = model
        self.amount_gas_used = amount_gas_used

    def execute(self, sim: Simulation) -> None:
        m = self.model

        if self.time > m.batch_times[m.current_batch]:
            m.save_batch_statistics(self.time - m.batch_times[m.current_batch])
            m.current_batch += 1

        if self.time > m.warmup_period: m.base_fee.update(self.time - m.batch_times[m.current_batch], m.b)
        #update next base fee
        m.b = max(m.b_min, m.b*(1 + 1 / 8 * (self.amount_gas_used - m.gas_target) / m.gas_target))
        if self.time > m.warmup_period: m.base_fee.update(self.time - m.batch_times[m.current_batch], m.b)

if __name__ == "__main__":
    import time
    start = time.time()
    model = GasMarket(12, 100_000, 90_000)
    model.run()
    model.report()
    print(f"Simulation ran for {(time.time() - start):.4f} seconds")
