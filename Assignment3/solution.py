from __future__ import annotations

import math
import numpy as np
import random
import os
import sys
import bisect
from tqdm import tqdm
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event
from des_library.statistics import _t_critical
from statistics_helper import *

class CTDepartment:
    def __init__(
        self,
        num_scanners: int = 2,
        num_scanners_night: int = 1,
        num_chairs: int = 3,
        morning_hourly_capacity: int = 4,
        afternoon_hourly_capacity: int = 3,
        do_schedule_outpatients: bool = True,
        do_schedule_inpatients: bool = True,
        do_schedule_emergency_patients: bool = True,
        service_time_distr: Callable = random.uniform,
        service_time_param: tuple = (10, 19),
        warmup_period: float = 0.0,
        num_batches: int = 50,
        stopping_time: float = 1e4,
        progress_bar: tqdm|None = None,
        seed: int = 42
        ):
        """
        Model a CT department with :num_scanners: scanners, :num_scanners_night: of those scanners available outside office hours,
        waiting room capacity of :num_chairs:. :morning_hourly_capacity: outpatients can be scheduled each hour from 8.00 to 12.00,
        :afternoon_hourly_capacity: outpatients can be scheduled each our from 12.00 to 16.00.
        """
        random.seed(seed)
        
        self.num_scanners: int = num_scanners
        self.num_scanners_night: int = num_scanners_night
        self.num_chairs: int = num_chairs
        self.morning_hourly_capacity: int = morning_hourly_capacity
        self.afternoon_hourly_capacity: int = afternoon_hourly_capacity
        self.do_schedule_outpatients: bool = do_schedule_outpatients
        self.do_schedule_inpatients: bool = do_schedule_inpatients
        self.do_schedule_emergency_patients: bool = do_schedule_emergency_patients
        
        self.service_time_distribution = service_time_distr
        self.service_time_param = service_time_param
        
        self.stopping_time = stopping_time
        self.num_batches = num_batches
        
        self.warmup_period = warmup_period
        self.batch_times = np.linspace(self.warmup_period, self.stopping_time, self.num_batches+1)
        self.current_batch: int = 0
        self.current_cycle: int = 0
        
        self.day_of_week: int = 0 # runs from 0 to 6 or from monday to sunday.
        self._last_update_hour: float = 0.0
        self._day_changed: bool = False
        self._last_progress_update: float = 0.0
        
        self._inpatient_is_traveling: bool = False

        self.sim = Simulation()

        # before event hooks
        self.sim.on_before_event(self._update_day)
        self.sim.on_before_event(self._weekly_reset_schedule)
        if progress_bar is not None:
            self.sim.on_before_event(lambda sim, event: self._update_progress(sim, event, progress_bar))
        
        # after event hooks
        self.sim.on_after_event(self._call_inpatient)
        self.sim.on_after_event(self._check_regen_condition)
        self.sim.on_after_event(self._check_batch)
        

        self.currently_scanning: list[Patient] = [] # patients that are currently being scanned. Can be at most :num_scanners:
        self.queue: list[Patient] = [] # patients currently waiting in waiting room
        self.inpatient_waiting_list: list[Patient] = [] # inpatients who have called today but do not find an empty room.
        
        self.hourly_schedule: dict[int, tuple[list[Patient], list[Patient]]] = {
            i: ([],[]) for i in range(5)
        } # maps day to tuple (morning schedule, afternoon schedule) for scheduling outpatients
        self.outpatient_waiting_list: list[Patient] = [] # outpatients who have called but have yet to be scheduled
        
        # arrival rates per minute (:time: is in minutes)
        self.outpatient_arrival_rate = 23/8 / 60
        self.inpatient_arrival_rate = lambda time: (3 / 8 + 81 * np.pi / 48 * abs(np.sin(np.pi/3*(time/(60) - 9)))) / 60 if 9 <= time / 60 <= 15 else (3 / 8) / 60
        self.emergency_arrival_rate = 1/60
        
        self.outpatient_show_probability: float = 0.84

        #statistics: keep all batch/regenerative statistics in a dict.
        self.statistics: dict[str, SampleBatchStatistic | TimeWeightedBatchStatistic | RateBatchStatistic | FractionBatchStatistic] = {
            "Waiting time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Inpatient waiting time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Outpatient waiting time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Emergency patient waiting time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Total fraction patients wait outside": FractionBatchStatistic(self.batch_times, warmup_period),
            "Fraction inpatients wait outside": FractionBatchStatistic(self.batch_times, warmup_period),
            "Fraction outpatients wait outside": FractionBatchStatistic(self.batch_times, warmup_period),
            "Fraction emergency patients wait outside": FractionBatchStatistic(self.batch_times, warmup_period),
            "Scanner utilization in office hours": TimeWeightedBatchStatistic(self.batch_times, warmup_period),
            "Scanner utilization outside office hours": TimeWeightedBatchStatistic(self.batch_times, warmup_period),
            "Queue size": TimeWeightedBatchStatistic(self.batch_times, warmup_period),
            "Waiting list size": TimeWeightedBatchStatistic(self.batch_times, warmup_period),
            "Outpatient access time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Inpatient access time": SampleBatchStatistic(self.batch_times, warmup_period),
            "Fraction inpatients scanned outside office hours": FractionBatchStatistic(self.batch_times, warmup_period),
        }
        self.num_outpatient_calls = Counter()
        self.num_outpatient_arrivals = Counter()
        self.num_inpatient_requests = Counter()
        self.num_inpatient_arrivals = Counter()
        self.num_emergency_arrivals = Counter()
        self.num_scanned_patients = Counter()

    def _update_day(self, sim: Simulation, event: Event):
            hour = (event.time / 60) % 24
            self._day_changed = False
            if hour < self._last_update_hour:  # hour wrapped around (crossed midnight)
                self.day_of_week = (self.day_of_week + 1) % 7
                self._day_changed = True
            
            self._last_update_hour = hour
            
    def _weekly_reset_schedule(self, sim: Simulation, event: Event):
            if self.day_of_week != 0 or not self._day_changed: return
            # at start of monday, empty schedule and schedule all waiting outpatients
            self.reset_schedule()
            remaining_patients = []
            for patient in self.outpatient_waiting_list:
                res = self.find_next_available_day(event.time)
                if res is None:
                    remaining_patients.append(patient)
                    continue
                capacity = self.morning_hourly_capacity if res[1] else self.afternoon_hourly_capacity
                self.schedule_outpatient(res[0], res[1], patient, event.time, sim, 60 * res[2] / capacity)
            self.outpatient_waiting_list = remaining_patients
            self.statistics["Waiting list size"].update(event.time, len(self.outpatient_waiting_list))
    
    def _call_inpatient(self, sim: Simulation, event: Event):
        if not (len([pat for pat in self.queue if pat.patient_type == "in"]) == 0 and not self._inpatient_is_traveling and self.inpatient_waiting_list): return
        patient = self.inpatient_waiting_list.pop(0)
        travel_time = random.uniform(9,15)
        self._inpatient_is_traveling = True
        sim.schedule(InpatientArrival(event.time + travel_time, self, patient))

    def _check_regen_condition(self, sim: Simulation, event: Event):
        condition = len(self.currently_scanning) == 0
        if condition: self.new_cycle(event.time)

    def _check_batch(self, sim: Simulation, event: Event):
        if event.time > self.batch_times[self.current_batch + 1]:
            self.new_batch(event.time)
    
    def _update_progress(self, sim: Simulation, event: Event, pbar: tqdm):
        pbar.update(event.time - self._last_progress_update)
        self._last_progress_update = event.time
    
    def new_cycle(self, now: float):
        self.current_cycle += 1
        for statistic in self.statistics.values():
            statistic.new_cycle(now)
            
    def new_batch(self, now: float):
        self.current_batch += 1
        for statistic in self.statistics.values():
            statistic.new_batch(now)
            
    def insert_patient(self, patient: Patient, now: float, sim: Simulation):
        is_full = len(self.queue) >= self.num_chairs
        self.statistics["Total fraction patients wait outside"].increment_total(now)
        if is_full: self.statistics["Total fraction patients wait outside"].increment(now)
        match patient.patient_type:
            case "out":
                self.statistics["Fraction outpatients wait outside"].increment_total(now)
                if is_full: self.statistics["Fraction outpatients wait outside"].increment(now)
            case "in":
                self.statistics["Fraction inpatients wait outside"].increment_total(now)
                if is_full: self.statistics["Fraction inpatients wait outside"].increment(now)
            case "emergency":
                self.statistics["Fraction emergency patients wait outside"].increment_total(now)
                if is_full: self.statistics["Fraction emergency patients wait outside"].increment(now)
        
        keys = [(pat.priority, pat.arrival_time) for pat in self.queue]
        idx = bisect.bisect_right(keys, (patient.priority, patient.arrival_time))
        self.queue.insert(idx, patient)

        is_weekend = self.day_of_week == 5 or self.day_of_week == 6
        is_office_hours: bool = 8 <= (now / 60) % 24 <= 16 and not is_weekend
        if is_office_hours:
            while self.queue and len(self.currently_scanning) < self.num_scanners:
                self.start_scanning(self.queue[0], now, sim)
            self.statistics["Scanner utilization in office hours"].update(now, len(self.currently_scanning) / self.num_scanners)
        else:
            while self.queue and len(self.currently_scanning) < self.num_scanners_night:
                self.start_scanning(self.queue[0], now, sim)
            self.statistics["Scanner utilization outside office hours"].update(now, len(self.currently_scanning) / self.num_scanners_night)

        self.statistics["Queue size"].update(now, len(self.queue))
        
    def recieve_inpatient_request(self, patient: Patient, now: float, sim: Simulation):
        patient.request_time = now
        keys = [pat.arrival_time for pat in self.inpatient_waiting_list]
        idx = bisect.bisect_right(keys, patient.arrival_time)
        self.inpatient_waiting_list.insert(idx, patient)

    def start_scanning(self, patient: Patient, now: float, sim: Simulation):
        self.queue.remove(patient)
        # update queue size after removal so time-weighted statistic records the decrease
        self.statistics["Queue size"].update(now, len(self.queue))
        self.currently_scanning.append(patient)

        self.statistics["Waiting time"].record(now, now - patient.arrival_time)
        match patient.patient_type:
            case "emergency":
                self.statistics["Emergency patient waiting time"].record(now, now - patient.arrival_time)
            case "in":
                self.statistics["Inpatient waiting time"].record(now, now - patient.arrival_time)
            case "out":
                self.statistics["Outpatient waiting time"].record(now, now - patient.arrival_time)
                
        # scanning_time = random.uniform(10,19)
        scanning_time = self.service_time_distribution(*self.service_time_param)
        sim.schedule(EndScan(now + scanning_time, self, patient))
        
    def reset_schedule(self):
        for day in self.hourly_schedule.keys():
            self.hourly_schedule[day] = ([],[])
            
    def find_next_available_day(self, now: float) -> tuple[int, bool, int] | None:
        '''
        Returns the next day an open schedule slot is available,
        with a boolean whether it is in the morning or not and the amount of people already scheduled that day.
        Returns None if no open slot is available this week.
        '''
        current_hour = math.ceil((now / 60) % 24)
        current_day = self.day_of_week
        
        hours_into_morning = max(min(current_hour - 8, 4), 0)
        hours_into_afternoon = max(min(current_hour - 12, 4), 0)
        
        open_mornings: set[tuple[int, int]] = set() # { (day, amount people scheduled that day), ...}
        open_afternoons: set[tuple[int, int]] = set()
        for day in self.hourly_schedule.keys():
            if day < current_day: continue
            if day > current_day: 
                hours_into_morning = 0
                hours_into_afternoon = 0
            morning_schedule, afternoon_schedule = self.hourly_schedule[day]
            if len(morning_schedule) < self.morning_hourly_capacity*(4 - hours_into_morning):
                open_mornings.add((day, len(morning_schedule)))
            if len(afternoon_schedule) < self.afternoon_hourly_capacity*(4 - hours_into_afternoon):
                open_afternoons.add((day, len(afternoon_schedule)))
        if len(open_mornings) == len(open_afternoons) == 0:
            return
        next_available_day, amount_people_scheduled = min(open_mornings.union(open_afternoons), key=lambda tup: (tup[0], 0 if tup in open_mornings else 1, tup[1]))
        return next_available_day, (next_available_day, amount_people_scheduled) in open_mornings, amount_people_scheduled
    
    def schedule_outpatient(self, day: int, is_morning: bool, patient: Patient, now: float, sim: Simulation,  minutes: float = 0.0):
        i = 0 if is_morning else 1
        self.hourly_schedule[day][i].append(patient)
        capacity = self.morning_hourly_capacity if is_morning else self.afternoon_hourly_capacity
        num = ((len(self.hourly_schedule[day][i]) - 1) // capacity)

        part_start = 8 + i * 4
        if day == self.day_of_week:
            current_minutes = now % (24 * 60)
            part_start_minutes = part_start * 60
            if current_minutes > part_start_minutes:
                elapsed = current_minutes - part_start_minutes
                earliest_slot = math.ceil(elapsed / 60)
                num = max(num, earliest_slot)

        arrival_time = ((day - self.day_of_week) * 24 + part_start + num) * 60 - (now % (24 * 60)) + minutes
        if arrival_time < 0:
            arrival_time += 24 * 60

        if random.random() < self.outpatient_show_probability:
            # outpatients show up with a probability p_s
            sim.schedule(OutpatientArrival(now + arrival_time, self, patient))
            
    def validate_num_batches(self, precision: float = 0.05, confidence: float = 0.95):
        r = self.num_batches
        quantile = _t_critical(1 - confidence / 2, r - 1)
        for stat_name, stat in self.statistics.items():
            value = quantile**2 * stat.batch_statistic.variance() / (precision / (1 + precision) * stat.batch_statistic.mean())**2
            if self.num_batches < value:
                print(f"{stat_name} did NOT PASS batch number validation")
            else:
                print(f"{stat_name} PASSED batch number validation")
            print(f"expression: {value:.4f}")

    def validate_warmup(self, precision: float = 0.05):
        if self.warmup_period == 0.0: raise ValueError("Cannot validate nonexistent warmup") 
        for stat_name, stat in self.statistics.items():
            expression = abs(stat.warmup_checks[1] / stat.warmup_checks[0] - 1)
            if expression > precision:
                print(f"{stat_name} did NOT PASS warmup validation")
            else:
                print(f"{stat_name} PASSED warmup validation")
            print(f"expression: {expression:.4f}")

    def run(self):
        if self.do_schedule_outpatients: self.sim.schedule(OutpatientCall(0.0, self))
        if self.do_schedule_inpatients: self.sim.schedule(InpatientRequestArrival(0.0, self))
        if self.do_schedule_emergency_patients: self.sim.schedule(EmergencyPatientArrival(0.0, self))
        self.sim.run(lambda sim: sim.current_time > self.stopping_time)
    
    def report(self):
        t = self.sim.current_time
        print("\nCTDepartment model")
        print(f"Horizon time: \t\t\t\t {t}")
        print(f"Number of outpatient calls: \t\t {self.num_outpatient_calls.value}")
        print(f"Number of outpatients arrived: \t\t {self.num_outpatient_arrivals.value}")
        print(f"Number of inpatient requests arrived: \t {self.num_inpatient_requests.value}")
        print(f"Number of inpatients arrived: \t\t {self.num_inpatient_arrivals.value}")
        print(f"Number of emergency patients arrived: \t {self.num_emergency_arrivals.value}")
        print(f"Num batches: \t\t\t\t {self.current_batch}")
        print(f"Num cycles: \t\t\t\t {self.current_cycle}")
        
        # Print statistics in a neatly aligned table
        max_name = max(len(name) for name in self.statistics)
        ci_width = 25
        header = (
            f"{'Statistic':<{max_name}}  {'Batch':>12}  {'95% CI':>{ci_width}}  "
            f"{'Regen':>12}  {'95% CI':>{ci_width}}  {'Full series':>12}"
        )
        print(header)
        for stat_name, stat in self.statistics.items():
            if stat.num_samples == 0: continue
            mean = stat.mean(t)
            conf_int = stat.confidence_interval()
            ci_batch = f"[{conf_int[0][0]:.4f}, {conf_int[0][1]:.4f}]"
            ci_regen = f"[{conf_int[1][0]:.4f}, {conf_int[1][1]:.4f}]"
            print(
                f"{stat_name:<{max_name}}  {mean[0]:12.4f}  {ci_batch:>{ci_width}}  "
                f"{mean[1]:12.4f}  {ci_regen:>{ci_width}}  {mean[2]:12.4f}"
            )

class Patient:
    def __init__(self, patient_type: str, arrival_time: float,):
        if patient_type != "in" and patient_type != "out" and patient_type != "emergency": raise ValueError("Invalid patient type")
        self.patient_type = patient_type
        self.arrival_time = arrival_time
        self.request_time: float|None = None
        self.priority = 0 if patient_type == "emergency" else 1 # emergency patients have priority 0, others have priority 1

class OutpatientCall(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation):
        m = self.model

        hour = (self.time / 60) % 24
        is_morning: bool = 8 <= hour <= 12
        is_afternoon: bool = 12 < hour <= 16
        is_weekend: bool = m.day_of_week == 5 or m.day_of_week == 6
        is_office_hours = (is_morning or is_afternoon) and not is_weekend
        if not is_office_hours:
            # outpatient calls do not arrive outside office hours, can do this because of exponential
            next_arrival_time = random.expovariate(m.outpatient_arrival_rate)
            sim.schedule(OutpatientCall(self.time + next_arrival_time, m))
            return
        
        m.num_outpatient_calls.increment()
        
        patient = Patient("out", self.time)
        res = m.find_next_available_day(self.time)
        if res is None: # did not find available time slot
            m.outpatient_waiting_list.append(patient)
            m.statistics["Waiting list size"].update(self.time, len(m.outpatient_waiting_list))
        else:
            next_available_day, is_scheduled_in_morning, amount_people_scheduled = res
            capacity = m.morning_hourly_capacity if is_scheduled_in_morning else m.afternoon_hourly_capacity
            m.schedule_outpatient(next_available_day, is_scheduled_in_morning, patient, self.time, sim, 60 * amount_people_scheduled / capacity)
                
        # schedule next arrival
        next_arrival_time = random.expovariate(m.outpatient_arrival_rate)
        sim.schedule(OutpatientCall(self.time + next_arrival_time, m))
        
class OutpatientArrival(Event):
    def __init__(self, time: float, model: CTDepartment, patient: Patient):
        super().__init__(time)
        self.model = model
        self.patient = patient
    
        
    def execute(self, sim: Simulation):
        m = self.model
        
        m.num_outpatient_arrivals.increment()
        m.statistics["Outpatient access time"].record(self.time, self.time - self.patient.arrival_time)
        
        self.patient.arrival_time = self.time
        m.insert_patient(self.patient, self.time, sim)
        

class InpatientRequestArrival(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model
        
    def execute(self, sim: Simulation):
        m = self.model
        
        m.num_inpatient_requests.increment()
        
        patient = Patient("in", self.time)
        m.recieve_inpatient_request(patient, self.time, sim)
        
        # schedule next event
        lamb_0 = (3 / 8 + 81*np.pi / 48) / 60
        arrival_time: float|None = None
        arrival_time = self.time
        while True: # thinning process
            tau = random.expovariate(lamb_0)
            arrival_time += tau
            if random.random() > m.inpatient_arrival_rate(arrival_time) / lamb_0: 
                continue
            break
        sim.schedule(InpatientRequestArrival(arrival_time, m))
        
class InpatientArrival(Event):
    def __init__(self, time: float, model: CTDepartment, patient: Patient):
        super().__init__(time)
        self.model = model
        self.patient = patient

    def execute(self, sim: Simulation):
        m = self.model
        assert self.patient.request_time is not None

        m.num_inpatient_arrivals.increment()
        m.statistics["Inpatient access time"].record(self.time, self.time - self.patient.arrival_time)

        self.patient.arrival_time = self.time
        m.insert_patient(self.patient, self.time, sim)
        m._inpatient_is_traveling = False

        if not (8 <= (self.patient.request_time / 60) % 24 <= 16 and m.day_of_week < 5): return
        m.statistics["Fraction inpatients scanned outside office hours"].increment_total(self.time)
        is_office_hours = 8 <= (self.time / 60) % 24 <= 16 and m.day_of_week < 5
        if not is_office_hours: m.statistics["Fraction inpatients scanned outside office hours"].increment(self.time)

class EmergencyPatientArrival(Event):
    def __init__(self, time: float, model: CTDepartment):
        super().__init__(time)
        self.model = model

    def execute(self, sim: Simulation):
        m = self.model
        
        m.num_emergency_arrivals.increment()
        
        patient = Patient("emergency", self.time)
        m.insert_patient(patient, self.time, sim)
        
        # schedule next arrival
        next_arrival_time = random.expovariate(m.emergency_arrival_rate)
        sim.schedule(EmergencyPatientArrival(self.time + next_arrival_time, m))

class EndScan(Event):
    # departures for when patients are done scanning.
    def __init__(self, time: float, model: CTDepartment, patient: Patient):
        super().__init__(time)
        self.model = model
        self.patient = patient
    
    def execute(self, sim: Simulation):
        m = self.model
        
        m.num_scanned_patients.increment()
        
        hour = self.time / 60 % 24
        is_weekend: bool = m.day_of_week == 5 or m.day_of_week == 6
        is_office_hours: bool = 8 <= hour <= 16 and not is_weekend

        m.currently_scanning.remove(self.patient)

        if m.queue:
            if is_office_hours and len(m.currently_scanning) < m.num_scanners:
                m.start_scanning(m.queue[0], self.time, sim)
            elif not is_office_hours and len(m.currently_scanning) < m.num_scanners_night:
                m.start_scanning(m.queue[0], self.time, sim)

        if is_office_hours:
            m.statistics["Scanner utilization in office hours"].update(self.time, len(m.currently_scanning) / m.num_scanners)
        else:
            m.statistics["Scanner utilization outside office hours"].update(self.time, len(m.currently_scanning) / m.num_scanners_night)

if __name__ == "__main__":
    stopping_time = 1e8
    # Verification cases:
    # compare M/G/1 model (service time B ~ U[10,19]) with simulated results: Wq = ~2.3846 min, Lq = ~0.0397, W = ~16.88 min, L = ~0.2814
    # M/G/2: Wq = 0.111, Lq = 0.00185
    c = 1
    # mgc_model = CTDepartment(num_scanners=c, num_scanners_night=c, do_schedule_inpatients=False, do_schedule_outpatients=False, stopping_time=stopping_time, progress_bar=tqdm(total=stopping_time, bar_format = "{desc}: {bar}| {percentage:.2f}% | [{elapsed}s]"))
    # mgc_model.run()
    # mgc_model.report()
    
    # M/M/1: Wq = 5.0, Lq = 0.08333...
    # M/M/2: Wq = 0.2381, L_1 = 0.00397
    # mmc_model = CTDepartment(num_scanners=c, num_scanners_night=c, service_time_distr=random.expovariate, service_time_param=(1/15,), do_schedule_inpatients=False, do_schedule_outpatients=False, stopping_time=stopping_time, progress_bar=tqdm(total=stopping_time, bar_format = "{desc}: {bar}| {percentage:.2f}% | [{elapsed}s]"))
    # mmc_model.run()
    # mmc_model.report()
    # mmc_model.validate_num_batches()
    
    model = CTDepartment(num_scanners=2, num_scanners_night=1, warmup_period=2e7, num_batches=50, stopping_time=stopping_time, progress_bar=tqdm(total=stopping_time, bar_format = "{desc}: {bar}| {percentage:.2f}% | [{elapsed}s]"))
    model.run()
    model.report()
    # model.validate_num_batches(precision=0.1)
    # model.validate_warmup()