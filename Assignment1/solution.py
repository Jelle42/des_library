
from __future__ import annotations

import bisect
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic

class Vehicle:
    def __init__(self, battery_level: float, arrival_time: float, patience_threshold: float):
        self.remaining = 60 * (1 - battery_level)
        self.arrival_time = arrival_time

class ChargingStationModel:
    def __init__(self):

        self.sim = Simulation()

class Arrival(Event):
    def __init__(self, time):
        super().__init__(time)

class Departure(Event):
    def __init__(self, time):
        super().__init__(time)