
from __future__ import annotations

import math
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

def arrival_time_function(n: int) -> float:
    return 15 * (1 + math.sin(n * math.pi / 12))**2 + 2

def battery_level_function(n: int) -> float:
    return 0.5 * abs(math.sin(n * math.pi / 7) + 1)

def patience_level_function(n: int) -> float:
    return 20 * (1 + abs(math.cos(n * math.e)))

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
        self.start_charging_time: float | None = None

        # pointers to events that sometimes need to be cancelled
        self.reneging_event = reneging_event
        self.departure_event = departure_event

    def decrease_remaining(self, amount: float) -> None:
        self.remaining -= amount

class ChargingStationModel:
    def __init__(self, num_chargers: int = 4, seed: int = 70):
        random.seed(seed)
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
        car.start_charging_time = now
        car.departure_event = Departure(now + car.remaining, self, car)
        self.sim.schedule(car.departure_event)
        self.charger_utilisation.update(now, min(self.num_chargers, len(self.queue)) / self.num_chargers)

    def run(self):
        self.sim.schedule(Arrival(0.0, self))
        
        def stopping_condition(sim: Simulation, model: ChargingStationModel = self, threshold: int = 800) -> bool:
            return model.completed_vehicles >= threshold
    
        self.sim.run(stop_condition=stopping_condition)
    
    def report(self):
        t = self.sim.current_time
        print("Charging station model")
        print(f"Horizon time: {t} min")
        print(f"Avg. queue length {self.queue_length.mean(t):.4f}")
        print(f"Avg. waiting time {self.waiting_time.mean():.4f}")
        print(f"Number of reneged vehicles {self.reneging_counter.value}")
        print(f"Reneging fraction {(self.reneging_counter.value / self.num_vehicles):.4f}")
        print(f"Avg. charger utilisation {self.charger_utilisation.mean(t)}")
        print(f"Number of early departures: {self.early_departure_counter.value}")
        print(f"Early departure fraction {(self.early_departure_counter.value / self.completed_vehicles):.4f}")
        print(f"Number of completed vehicles: {self.completed_vehicles}")
        print(f"Number of arrived vehicles: {self.num_vehicles}")

class Arrival(Event):
    def __init__(self, time, model: ChargingStationModel):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation) -> None:
        m = self.model
        m.queue_length.update(self.time, max(len(m.queue) - 4, 0))

        battery_level = battery_level_function(m.num_vehicles)
        patience_threshold = patience_level_function(m.num_vehicles)

        new_car = Vehicle(battery_level, self.time) # create new vehicle object

        queue_length = len(m.queue)

        for i in range(min(queue_length, m.num_chargers)):
            current_car = m.queue[i]
            assert current_car.start_charging_time is not None
            current_car.decrease_remaining(self.time - current_car.start_charging_time)

        if queue_length < m.num_chargers:
            m.start_charging(self.time, new_car) # if chargers available, start charging
        else:
            if queue_length % 5 == 0: # if queue length is a multiple of 5, check for early departures
                for car in m.queue:
                    if car.remaining > 15: continue # only check cars that have a remaining time of less than 15 minutes
                    if random.random() > 0.2: continue # cars have a probability of 0.2 to leave early
                    if car.departure_event:
                        if car.departure_event.is_early: continue # do not cancel/override other early departures
                        car.departure_event.cancel()
                    early_departure = Departure(self.time + 2, m, car, True)
                    car.departure_event = early_departure
                    sim.schedule(early_departure)

            reneging_event = Renege(self.time + patience_threshold, m, new_car)
            new_car.reneging_event = reneging_event
            m.sim.schedule(reneging_event) # schedule reneging

        m.insert_vehicle(new_car) # add vehicle to queue

        next_arrival_time = arrival_time_function(m.num_vehicles)

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
        m.queue_length.update(self.time, max(len(m.queue) - 4, 0))

        #actually remove car from queue and start charging next car
        if self.car in m.queue:
            m.queue.remove(self.car)
        m.queue_length.update(self.time, len(m.queue))
        m.waiting_time.record(self.time - self.car.arrival_time)

        if len(m.queue) >= m.num_chargers:
            m.start_charging(self.time, m.queue[m.num_chargers - 1]) # start charging next person
        m.completed_vehicles += 1
        if self.is_early: m.early_departure_counter.increment()

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
        m.reneging_counter.increment()

if __name__ == "__main__":
    model = ChargingStationModel(4)
    model.run()
    model.report()