from __future__ import annotations

# import math
import numpy as np
import random
import os
import sys
import bisect
# from scipy.stats import t

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class CTDepartment:
    def __init__(self, num_scanners: int, num_chairs: int = 3):
        self.num_scanners = num_scanners
        self.num_chairs = num_chairs

        self.sim = Simulation()

        self.queue: list[Patient] = []

        #statistics
        self.waiting_time = SampleStatistic()

class Patient:
    def __init__(self, priority: int):
        self.priority = priority
    # use one class for all patient types, emergency patients get higher priority. use this when inserting into queue.

class Arrival(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model

        #Two options: 
        # -One arrival loop and determine probabilistically what type of patient it is
        # -Or three arrival loops. idk what is cleaner

class Departure(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model
    
    # departures for when patients are done scanning.
