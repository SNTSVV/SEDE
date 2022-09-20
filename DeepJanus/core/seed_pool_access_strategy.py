import random

from DeepJanus.core.config import Config
from DeepJanus.core.individual import Individual
from DeepJanus.core.seed_pool import SeedPool


class SeedPoolAccessStrategy:
    def __init__(self, pool: SeedPool):
        self.pool = pool
        self.counter = -1

    def get_member_randomly(self) -> Individual:
        return random.choice(self.pool)

    def get_seed(self):
        generator_name = self.pool.problem.config.generator_name
        if generator_name == Config.GEN_RANDOM:
            seed = self.pool.problem.generate_random_member()
        elif generator_name == Config.GEN_RANDOM_SEEDED:
            seed = self.get_member_randomly()
        elif generator_name == Config.GEN_SEQUENTIAL_SEEDED:
            seed = self.get_member_circular()
        else:
            raise NotImplemented(generator_name)
        return seed