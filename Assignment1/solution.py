
from __future__ import annotations

import math
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class Vehicle:
    def __init__(
            self,
            battery_level: float,
            arrival_time: float,
            reneging_event: Renege | None = None,
            departure_event: Departure | None = None,
        ):
        self.remaining = 60 * (1 - battery_level)
        self.arrival_time = arrival_time
        # pointers to events that sometimes need to be cancelled
        self.reneging_event = reneging_event
        self.departure_event = departure_event

class ChargingStationModel:
    def __init__(self, num_chargers: int = 4):
        self.num_chargers = num_chargers
        self.num_vehicles = 0 # used for the n in calculating arrival times, battery levels and patience thresholds.
        self.completed_vehicles = 0 # count number of completed vehicles
        self.sim = Simulation()
        self.queue: list[Vehicle] = []

        # statistics to keep track of
        self.queue_length = TimeWeightedStatistic()
        self.waiting_time = SampleStatistic()
        self.reneging_counter = Counter()
        self.charger_utilisation = TimeWeightedStatistic()
        self.early_departure_counter = Counter()

    def insert_vehicle(self, car: Vehicle) -> None:
        self.queue.append(car)
        self.num_vehicles += 1

    def start_charging(self, now: float, car: Vehicle) -> None:
        if car.reneging_event:
            car.reneging_event.cancel()
        car.departure_event = Departure(now + car.remaining, self, car)
        self.sim.schedule(car.departure_event)

    def run(self):
        self.sim.schedule(Arrival(0.0, self))
        
        def stopping_condition(sim: Simulation, model: ChargingStationModel = self, threshold: int = 800) -> bool:
            return model.completed_vehicles >= threshold
    
        self.sim.run(stop_condition=stopping_condition)


class Arrival(Event):
    def __init__(self, time, model: ChargingStationModel):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.queue_length.update(self.time, len(m.queue))

        battery_level = 0.5 * abs(math.sin(m.num_vehicles * math.pi / 7) + 1)
        patience_threshold = 20 * (1 + abs(math.cos(m.num_vehicles * math.e)))

        new_car = Vehicle(battery_level, self.time) # create new vehicle object

        queue_length = len(m.queue)
        if queue_length < 4:
            m.start_charging(self.time, new_car)
        else:
            if queue_length % 5 == 0:
                for car in m.queue:
                    if random.randint(1,5) != 1: continue
                    if car.departure_event:
                        if car.departure_event.is_early: continue # do not cancel early departures
                        car.departure_event.cancel()
                    early_departure = Departure(self.time + 2, m, car, True)
                    car.departure_event = early_departure
                    
            reneging_event = Renege(self.time + patience_threshold, m, new_car)
            new_car.reneging_event = reneging_event
            m.sim.schedule(reneging_event) # schedule renege

        m.insert_vehicle(new_car) # add vehicle to queue

        next_arrival_time = 15 * (1 + math.sin(m.num_vehicles * math.pi / 12))**2 + 2

        sim.schedule(Arrival(self.time + next_arrival_time, m)) # schedule new arrival

class Departure(Event):
    def __init__(self, time, model: ChargingStationModel, car: Vehicle, is_early: bool = False):
        super().__init__(time)
        self.model = model
        self.car = car
        self.is_early = is_early
    
    def execute(self, sim: Simulation) -> None:
        if self.cancelled:
            return
        m = self.model
        #update queue statistic
        m.queue_length.update(self.time, len(m.queue))

        #actually remove car from queue and start charging next car
        m.queue.remove(self.car)
        m.start_charging(self.time, m.queue[m.num_chargers - 1])
        m.completed_vehicles += 1

class Renege(Event):
    def __init__(self, time, model: ChargingStationModel, car: Vehicle):
        super().__init__(time)
        self.model = model
        self.car = car
    
    def execute(self, sim: Simulation) -> None:
        if self.cancelled:
            return
        m = self.model
        if self.car in m.queue:
            m.queue.remove(self.car)
        else:
            return
        m.completed_vehicles += 1
