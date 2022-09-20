from typing import List

from DeepJanus.core.config import Config
from DeepJanus.core.archive import Archive
from DeepJanus.core.individual import Individual
#from DeepJanus.core.seed_pool_impl import SeedPoolRandom

class Problem:
    def __init__(self, config: Config, archive: Archive, continueFlag: bool, caseFile):
        self.config: Config = config
        self.archive = archive
        self.continueFlag = continueFlag

    def deap_generate_individual(self) -> Individual:
        raise NotImplemented()

    def deap_mutate_individual(self, individual: Individual):
        individual.mutate()

    def deap_evaluate_individual(self, individual: Individual):
        raise NotImplemented()

    def deap_individual_class(self):
        raise NotImplemented()

    def on_iteration(self, idx, pop: List[Individual], logbook):
        raise NotImplemented()

    def member_class(self):
        raise NotImplemented()

    def reseed(self, population, offspring):
        raise NotImplemented()

    def population_metrics(self, pop, idx):
        raise NotImplemented()

    def archive_metrics(self, pop, idx):
        raise NotImplemented()

    def generate_random_member(self) -> Individual:
        raise NotImplemented()

    def pre_evaluate_members(self, individuals: List[Individual]):
        pass