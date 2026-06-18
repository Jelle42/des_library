from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import TimeWeightedStatistic, SampleStatistic, Counter

class TimeWeightedBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray, warmup_period: float = 0.0):
        self.running_statistic_batch = TimeWeightedStatistic()
        self.running_statistic_regen = TimeWeightedStatistic()
        self.running_statistic_full_series = TimeWeightedStatistic()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.warmup_period = warmup_period
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
    
    def update(self, current_time: float, new_value: float):
        self.num_samples += 1
        self.running_statistic_full_series.update(current_time, new_value)
        if current_time < self.warmup_period: return
        self.running_statistic_batch.update(current_time - self.batch_times[self.current_batch], new_value)
        self.running_statistic_regen.update(current_time - self.last_cycle_update, new_value)
        
    def record(self, current_time: float, new_value: float):
        raise NotImplementedError("Class TimeWeightedBatchStatistic does not have a method 'record'")
        
    def increment(self, current_time: float):
        raise NotImplementedError("Class TimeWeightedBatchStatistic does not have a method 'increment'")
    
    def increment_total(self, current_time: float):
        raise NotImplementedError("Class TimeWeightedBatchStatistic does not have a method 'increment_total'")
    
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_statistic_batch.mean(current_time - self.batch_times[self.current_batch]))
        self.running_statistic_batch.reset()
        self.current_batch += 1
        
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_statistic_regen.mean(current_time - self.last_cycle_update))
        self.running_statistic_regen.reset()
        self.current_cycle += 1
        self.last_cycle_update = current_time
    
    def mean(self, now: float) -> tuple[float, float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean(), self.running_statistic_full_series.mean(now)
    
    def confidence_interval(self, alpha: float = 0.95) -> tuple[tuple[float,float], tuple[float,float]]:
        return self.batch_statistic.confidence_interval(alpha), self.regen_statistic.confidence_interval(alpha)
        
class SampleBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray, warmup_period: float = 0.0):
        self.running_statistic_batch = SampleStatistic()
        self.running_statistic_regen = SampleStatistic()
        self.running_statistic_full_series = SampleStatistic()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.warmup_period = warmup_period
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
        
    def record(self, current_time: float, new_value: float):
        self.num_samples += 1
        self.running_statistic_full_series.record(new_value)
        if current_time < self.warmup_period: return
        self.running_statistic_batch.record(new_value)
        self.running_statistic_regen.record(new_value)
        
    def update(self, current_time: float, new_value: float):
        raise NotImplementedError("Class SampleBatchStatistic does not have a method 'update'")
        
    def increment(self, current_time: float):
        raise NotImplementedError("Class SampleBatchStatistic does not have a method 'increment'")
    
    def increment_total(self, current_time: float):
        raise NotImplementedError("Class SampleBatchStatistic does not have a method 'increment_total'")
        
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_statistic_batch.mean())
        self.running_statistic_batch.reset()
        self.current_batch += 1
    
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_statistic_regen.mean())
        self.running_statistic_regen.reset()
        self.current_cycle += 1
        self.last_cycle_update = current_time
        
    def mean(self, now: float) -> tuple[float, float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean(), self.running_statistic_full_series.mean()
    
    def confidence_interval(self, alpha: float = 0.95) -> tuple[tuple[float,float], tuple[float,float]]:
        return self.batch_statistic.confidence_interval(alpha), self.regen_statistic.confidence_interval(alpha)
        
class RateBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray, warmup_period: float = 0.0):
        self.running_counter_batch = Counter()
        self.running_counter_regen = Counter()
        self.running_counter_full_series = Counter()
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.warmup_period = warmup_period
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
        
    def increment(self, current_time: float, n: int = 1):
        self.num_samples += 1
        self.running_counter_full_series.increment(n)
        if current_time < self.warmup_period: return
        self.running_counter_batch.increment(n)
        self.running_counter_regen.increment(n)
    
    def update(self, current_time: float, new_value: float):
        raise NotImplementedError("Class RateBatchStatistic does not have a method 'update'")
        
    def record(self, current_time: float, new_value: float):
        raise NotImplementedError("Class RateBatchStatistic does not have a method 'record'")
    
    def increment_total(self, current_time: float):
        raise NotImplementedError("Class RateBatchStatistic does not have a method 'increment_total'")
        
    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_counter_batch.rate(current_time - self.batch_times[self.current_batch]))
        self.running_counter_batch.reset()
        self.current_batch += 1
        
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_counter_regen.rate(current_time - self.last_cycle_update))
        self.running_counter_regen.reset()
        self.current_cycle += 1
        self.last_cycle_update = current_time
        
    def mean(self, now: float) -> tuple[float, float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean(), self.running_counter_full_series.rate(now)
    
    def confidence_interval(self,  alpha: float = 0.95) -> tuple[tuple[float,float], tuple[float,float]]:
        return self.batch_statistic.confidence_interval(alpha), self.regen_statistic.confidence_interval(alpha)
    
class FractionBatchStatistic:
    def __init__(self, batch_times: list[float]|np.ndarray, warmup_period: float = 0.0):
        self.running_counter_batch = Counter()
        self.running_counter_regen = Counter()
        self.running_counter_full_series = Counter()
        
        self.running_total_batch = Counter()
        self.running_total_regen = Counter()
        self.running_total_full_series = Counter()
        
        self.batch_statistic = SampleStatistic()
        self.regen_statistic = SampleStatistic()
        
        self.batch_times = batch_times
        self.warmup_period = warmup_period
        self.last_cycle_update: float = 0.0
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.num_samples: int = 0
        
    def increment(self, current_time: float, n: int = 1):
        self.running_counter_full_series.increment(n)
        self.num_samples += 1
        if current_time < self.warmup_period: return
        self.running_counter_batch.increment(n)
        self.running_counter_regen.increment(n)
    
    def increment_total(self, current_time: float, n: int = 1):
        self.running_total_full_series.increment(n)
        if current_time < self.warmup_period: return
        self.running_total_batch.increment(n)
        self.running_total_regen.increment(n)
        
    def update(self, current_time: float, new_value: float):
        raise NotImplementedError("Class FractionBatchStatistic does not have a method 'update'")
        
    def record(self, current_time: float, new_value: float):
        raise NotImplementedError("Class FractionBatchStatistic does not have a method 'record'")

    def new_batch(self, current_time: float):
        self.batch_statistic.record(self.running_counter_batch.fraction(self.running_total_batch.value))
        self.running_counter_batch.reset()
        self.running_total_batch.reset()
        self.current_batch += 1
        
    def new_cycle(self, current_time: float):
        self.regen_statistic.record(self.running_counter_regen.fraction(self.running_total_regen.value))
        self.running_counter_regen.reset()
        self.running_total_regen.reset()
        self.current_cycle += 1
        self.last_cycle_update = current_time
        
    def mean(self, now: float) -> tuple[float, float, float]:
        return self.batch_statistic.mean(), self.regen_statistic.mean(), (self.running_counter_full_series.value / self.running_total_full_series.value if self.running_total_full_series.value != 0 else 0)
    
    def confidence_interval(self, alpha: float = 0.95) -> tuple[tuple[float,float], tuple[float,float]]:
        return self.batch_statistic.confidence_interval(alpha), self.regen_statistic.confidence_interval(alpha)