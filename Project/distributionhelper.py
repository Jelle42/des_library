"""
Probability distribution wrappers for stochastic simulations.

Each distribution exposes a ``sample()`` method (also callable via ``()``).
These are thin wrappers around ``random`` that keep simulation code
declarative::

    service = Exponential(mean=2.0)
    next_service_time = service()
"""
import math
import random
from typing import Callable


class Distribution:
    """Abstract base for all distributions."""

    def sample(self) -> float:
        raise NotImplementedError
    
    @property
    def mean(self) -> float:
        raise NotImplementedError
    
    @property
    def variance(self) -> float:
        raise NotImplementedError
    
    @property
    def std(self) -> float:
        raise NotImplementedError

    def __call__(self) -> float:
        return self.sample()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Deterministic(Distribution):
    """Always returns the same constant value."""

    def __init__(self, value: float):
        self.value = value

    def sample(self) -> float:
        return self.value
    
    @property
    def mean(self) -> float:
        return self.value
    
    @property
    def variance(self) -> float:
        return 0.0
    
    @property
    def std(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"Deterministic({self.value})"


class Exponential(Distribution):
    """Exponential distribution with given *mean*."""

    def __init__(self, mean: float):
        self.avg = mean

    def sample(self) -> float:
        return random.expovariate(1.0 / self.avg)
    
    @property
    def mean(self) -> float:
        return self.avg
    
    @property
    def variance(self) -> float:
        return self.avg**2
    
    @property
    def std(self) -> float:
        return self.avg

    def __repr__(self) -> str:
        return f"Exponential(mean={self.avg})"


class Erlang(Distribution):
    """Erlang-*k* distribution with given *mean*.

    An Erlang-k(mean) is the sum of k independent Exp(mean/k) variables.
    """

    def __init__(self, k: int, mean: float):
        self.k = k
        self.avg = mean
        self._rate = k / mean  # rate of each exponential phase

    def sample(self) -> float:
        return sum(random.expovariate(self._rate) for _ in range(self.k))
    
    @property
    def mean(self) -> float:
        return self.avg
    
    @property
    def variance(self) -> float:
        return self.mean**2 / self.k
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)        

    def __repr__(self) -> str:
        return f"Erlang(k={self.k}, mean={self.avg})"


class Uniform(Distribution):
    """Continuous uniform distribution on [*low*, *high*]."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low = low
        self.high = high

    def sample(self) -> float:
        return random.uniform(self.low, self.high)
    
    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2
    
    @property
    def variance(self) -> float:
        return (self.high - self.low)**2 / 12
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def __repr__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


class Normal(Distribution):
    """Normal (Gaussian) distribution with given *mean* and *sd*."""

    def __init__(self, mean: float = 0.0, sd: float = 1.0):
        self.mu = mean
        self.sd = sd

    def sample(self) -> float:
        return random.gauss(self.mu, self.sd)
    
    @property
    def mean(self) -> float:
        return self.mu
    
    @property
    def variance(self) -> float:
        return self.sd **2
    
    @property
    def std(self) -> float:
        return self.sd

    def __repr__(self) -> str:
        return f"Normal(mean={self.mu}, sd={self.sd})"
    
class Gamma(Distribution):
    """Gamma distribution with parameters *alpha* and *beta*"""
    def __init__(self, shape: float, scale: float) -> None:
        self.alpha = shape
        self.beta = 1/scale
        
    def sample(self) -> float:
        return random.gammavariate(self.alpha, self.beta)
    
    @property
    def mean(self) -> float:
        return self.alpha / self.beta
    
    @property
    def variance(self) -> float:
        return self.alpha / self.beta**2
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


class Sequence(Distribution):
    """Deterministic sequence driven by a function of *n*.

    ``func(n)`` is called with *n* = 0, 1, 2, … on successive samples.
    Call ``reset()`` to restart from *n* = 0.
    """

    def __init__(self, func: Callable[[int], float]):
        self.func = func
        self.n: int = 0

    def sample(self) -> float:
        value = self.func(self.n)
        self.n += 1
        return value

    def reset(self) -> None:
        self.n = 0

    def __repr__(self) -> str:
        return f"Sequence(n={self.n})"
