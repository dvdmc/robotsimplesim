import numpy as np
from abc import ABC, abstractmethod

class BasePrior(ABC):
    @abstractmethod
    def __call__(self, x):
        ...

class ConstantPrior(BasePrior):
    def __init__(self, value):
        self.value = value
    def __call__(self, x):
        return self.value
    
class DistanceBasedPrior(BasePrior):
    def __init__(self, origin, max_value, max_distance):
        self.origin = origin
        self.max_value = max_value
        self.max_distance = max_distance

    def __call__(self, x):
        distance = np.linalg.norm(x - self.origin)
        if distance > self.max_distance:
            return self.max_value
        else:
            return 1 + (self.max_value-1) * distance**2 / self.max_distance**2
        