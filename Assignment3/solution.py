from __future__ import annotations

# import math
import numpy as np
import random
import os
import sys
import bisect
# from scipy.stats import t

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event
from statistics_helper import *

class CTDepartment:
    def __init__(self, num_scanners: int, num_chairs: int = 3):
        self.num_scanners = num_scanners
        self.num_chairs = num_chairs

        self.sim = Simulation()

        self.queue: list[Patient] = []

        #statistics: keep all statistics in a dict.
        self.statistics: dict[str, SampleBatchStatistic|TimeWeightedBatchStatistic|RateBatchStatistic] = {
            "Waiting time": SampleBatchStatistic(),
        }
    
    def new_batch(self, now: float):
        for statistic in self.statistics.values():
            statistic.new_batch(now)
    
    def new_cycle(self, now: float):
        for statistic in self.statistics.values():
            statistic.new_cycle(now)

class Patient:
    def __init__(self, type: int, priority: int):
        self.patient_type = type # 0: emergency patient, 1: inpatient, 2: outpatient
        self.priority = priority
    # use one class for all patient types, emergency patients get higher priority. use this when inserting into queue.

class Arrival(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model

        #Two options: 
        # -One arrival loop and determine probabilistically what type of patient it is
        # -Or three arrival loops. idk what is cleaner
        # three loops is probably easier i think, since we have a non-homogeneous poisson process

class Departure(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model
    
    # departures for when patients are done scanning.
