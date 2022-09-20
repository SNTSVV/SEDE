from DeepJanus.core.problem import Problem
from DeepJanus.core.individual import Individual


class SeedPool:
    def __init__(self, problem: Problem):
        self.problem = problem

    def __len__(self):
        raise NotImplemented()

    def __getitem__(self, item) -> Individual:
        raise NotImplemented()
