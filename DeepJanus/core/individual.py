from typing import Tuple

from numpy import mean
import numpy as np

class Individual:
    def __init__(self):
        self.members_distance: float = None
        self.F2: float = None
        self.seed: Individual = None

    def clone(self) -> 'creator.base':
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()

    def mutate(self):
        raise NotImplemented()

    def semantic_distance(self, i2: 'Individual'):
        raise NotImplemented()

    def members_by_sign(self):
        result = self.members_by_distance_to_boundary()
        return result

    def members_by_distance_to_boundary(self):
        result = self
        return result