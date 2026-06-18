from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import TimeWeightedStatistic, SampleStatistic, Counter

class TimeWeightedBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray):
        self.running_statistic_batch = TimeWeightedStatistic()
        self.running_statistic_regen = TimeWeightedStatistic()
        self.running_statistic_full_series = TimeWeightedStatistic()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
    
    def update(self, current_time: float, new_value: float):
        self.running_statistic_batch.update(current_time - self.batch_times[self.current_batch], new_value)
        self.running_statistic_regen.update(current_time - self.last_cycle_update, new_value)
        self.running_statistic_full_series.update(current_time - self.batch_times[self.current_batch], new_value)
        self.num_samples += 1
    
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_statistic_batch.mean(current_time - self.batch_times[self.current_batch]))
        self.running_statistic_batch.reset()
        self.current_batch += 1
        
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_statistic_regen.mean(current_time - self.last_cycle_update))
        self.running_statistic_regen.reset()
        self.current_cycle += 1
        self.last_cycle_update = current_time
    
    def mean(self) -> tuple[float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean()
        
class SampleBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray):
        self.running_statistic_batch = SampleStatistic()
        self.running_statistic_regen = SampleStatistic()
        self.running_statistic_full_series = SampleStatistic()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
        
    def update(self, current_time: float, new_value: float):
        self.running_statistic_batch.record(new_value)
        self.running_statistic_regen.record(new_value)
        self.running_statistic_full_series.record(new_value)
        self.num_samples += 1
        
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_statistic_batch.mean())
        self.running_statistic_batch.reset()
        self.current_batch += 1
    
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_statistic_regen.mean())
        self.running_statistic_regen.reset()
        
    def mean(self) -> tuple[float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean()
        
class RateBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray):
        self.running_counter_batch = Counter()
        self.running_counter_regen = Counter()
        self.running_counter_full_series = Counter()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
        
    def update(self, current_time: float, n: int | float = 1):
        if isinstance(n, float):
            raise ValueError("n should be an int")
        self.running_counter_batch.increment(n)
        self.running_counter_regen.increment(n)
        self.running_counter_full_series.increment(n)
        self.num_samples += 1
        
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_counter_batch.rate(current_time - self.batch_times[self.current_batch]))
        self.running_counter_batch.reset()
        self.current_batch += 1
        
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_counter_regen.rate(current_time - self.last_cycle_update))
        self.running_counter_regen.reset()
        self.last_cycle_update = current_time
        
    def mean(self) -> tuple[float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean()