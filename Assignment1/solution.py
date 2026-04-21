
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class Vehicle:
    def __init__(self, battery_level: float, arrival_time: float, patience_threshold: float):
        self.remaining = 60 * (1 - battery_level)
        self.arrival_time = arrival_time

class ChargingStationModel:
    def __init__(self, num_chargers: int = 4):
        self.num_vehicles = 0
        self.sim = Simulation()

        #statistics to keep track of
        self.queue_length = TimeWeightedStatistic()
        self.waiting_time = SampleStatistic()
        self.reneging_counter = Counter()
        self.charger_utilisation = TimeWeightedStatistic()
        self.early_departure_counter = Counter()

    def run(self):
        self.sim.schedule(Arrival(0.0, self))

class Arrival(Event):
    def __init__(self, time, model: ChargingStationModel):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        battery_level = 0.5 * abs(math.sin(m.num_vehicles * math.pi /7) + 1)


        arrival_time = 15 * (1 + math.sin(m.num_vehicles * math.pi / 12))**2 + 2

        sim.schedule(Arrival(self.time + arrival_time, m))

class Departure(Event):
    def __init__(self, time, model: ChargingStationModel):
        super().__init__(time)
        self.model = model