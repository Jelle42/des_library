from __future__ import annotations

import math
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from des_library import Simulation, Event, TimeWeightedStatistic, SampleStatistic, Counter

class GasMarket:
    def __init__(self, gas_target: float, gas_limit: float, seed: int = 42):
        random.seed(seed)
        self.gas_target = gas_target
        self.gas_limit = gas_limit