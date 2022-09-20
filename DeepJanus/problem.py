
import random
from typing import List

from deap import creator

from searchModule import doParamDist, doImage
from DeepJanus.core.archive import Archive
from DeepJanus.core.folders import folders
from DeepJanus.core.log_setup import get_logger
from DeepJanus.core.metrics import get_radius_seed, get_diameter
from DeepJanus.core.misc import delete_folder_recursively
from DeepJanus.core.problem import Problem
from DeepJanus.core.seed_pool_access_strategy import SeedPoolAccessStrategy
from DeepJanus.core.seed_pool_impl import SeedPoolRandom
from DeepJanus.SEDE_config import SEDEConfig
from DeepJanus.individual import SEDEIndividual

log = get_logger(__file__)


class SEDEProblem(Problem):
    def __init__(self, config: SEDEConfig, archive: Archive, continueFlag: bool, caseFile):
        self.config: SEDEConfig = config
        super().__init__(config, archive, continueFlag, caseFile)
        #if not continueFlag:
        seed_pool = SeedPoolRandom(self, config.POPSIZE)
        self._seed_pool_strategy = SeedPoolAccessStrategy(seed_pool)
        self.experiment_path = folders.experiments.joinpath(self.config.experiment_name)
        delete_folder_recursively(self.experiment_path)
        self.distances = {}
        self.in_cluster = {}
        self.in_cluster_percentage = {}
        self.avg_dist = {}
        self.imgs = {}
        self.centroid_distances = {}

    def deap_generate_individual(self):
        seed = self._seed_pool_strategy.get_seed()
        individual = seed.clone().mutate()
        #individual: SEDEIndividual = creator.Individual(params, self.config, self.archive)
        #individual = SEDEIndividual(ind.params, centroid, self.config, self.archive)
        individual.seed = seed
        #log.info(f'generated {individual}')

        return individual

    def deap_evaluate_individual(self, individual: SEDEIndividual):
        return individual.evaluate()

    def on_iteration(self, idx, pop: List[SEDEIndividual], logbook):
        #self.archive.process_population(pop)
        return
        #self.experiment_path.mkdir(parents=True, exist_ok=True)
        #self.experiment_path.joinpath('config.json').write_text(json.dumps(self.config.__dict__))

        #gen_path = self.experiment_path.joinpath(f'gen{idx}')
        #gen_path.mkdir(parents=True, exist_ok=True)

        # Generate final report at the end of the last iteration.
        #if idx + 1 == self.config.NUM_GENERATIONS:
        #    report = {
        #        'archive_len': len(self.archive),
        #        'radius': get_radius_seed(self.archive),
        #        'diameter_out': get_diameter([ind.members_by_sign()[0] for ind in self.archive]),
        #        'diameter_in': get_diameter([ind.members_by_sign()[1] for ind in self.archive])
        #    }
        #    gen_path.joinpath(f'report{idx}.json').write_text(json.dumps(report))

        #BeamNGIndividualSetStore(gen_path.joinpath('population')).save(pop)
        #BeamNGIndividualSetStore(gen_path.joinpath('archive')).save(self.archive)

    def generate_random_member(self) -> SEDEIndividual:
        params = []
        for xl, xu in zip(self.config.lower_bound, self.config.upper_bound):
            params.append(random.uniform(xl, xu))
        individual = SEDEIndividual(params, self.config, self.archive)
        return individual

    def deap_individual_class(self):
        return SEDEIndividual

    def reseed(self, pop, offspring):
        if len(self.archive) > 0:
            stop = self.config.RESEED_UPPER_BOUND + 1
            seed_range = min(random.randrange(0, stop), len(pop))
            #log.info(f'reseed{seed_range}')
            #for i in range(0, seed_range):
            #    ind1 = self.deap_generate_individual()
            #    rem_idx = -(i + 1)
            #    log.info(f'reseed rem {pop[rem_idx]}')
            #    pop[rem_idx] = ind1
            archived_seeds = [i.seed for i in self.archive]
            for i in range(len(pop)):
                if pop[i].seed in archived_seeds:
                    ind1 = self.deap_generate_individual()
                    #log.info(f'reseed rem {pop[i]}')
                    pop[i] = ind1
    def archive_metrics(self, pop, idx):
        import os, shutil
        log.info("## archive metrics")
        self.distances[str(idx)] = []
        self.avg_dist[str(idx)] = []
        self.in_cluster[str(idx)] = 0
        self.in_cluster_percentage[str(idx)] = 0.0
        self.imgs = []
        self.centroid_distances[str(idx)] = []

        outPath = os.path.join(self.problemPath, str(idx))
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        in_cluster_nonduplicate = 0
        if len(self.archive) > 0:
            pop = self.archive
            candidateList = []
            fitnessList = []
            final_population = []
            for candidate in pop:
                candidateList.append(candidate)
                fitnessList.append(candidate.F1)
            if len(fitnessList) > 0:
                POPSIZE = 25 if len(fitnessList) > 25 else len(fitnessList)
                for i in range(0, POPSIZE):
                    indx = fitnessList.index(max(fitnessList))
                    final_population.append(candidateList[indx])
                    fitnessList.remove(fitnessList[indx])
                    candidateList.remove(candidateList[indx])
                pop = final_population
        for ind in pop:
            if ind.inCluster:
                if ind.export(outPath):
                    in_cluster_nonduplicate += 1
        shutil.rmtree(outPath)
        for ind in pop:
            self.centroid_distances[str(idx)].append(ind.F2)
            if ind.inCluster:
                self.imgs.append(os.path.basename(ind.imgPath))
                for ind2 in pop:
                    if ind2.inCluster:
                        param_dist = doParamDist(ind.params, ind2.params, self.config.upper_bound,
                                                 self.config.lower_bound)
                        if param_dist != 0:
                            self.distances[str(idx)].append(param_dist)

        self.in_cluster[str(idx)] = in_cluster_nonduplicate
        self.in_cluster_percentage[str(idx)] = 100 * (self.in_cluster[str(idx)] / 25)
        if len(self.distances[str(idx)]) > 0:
            self.avg_dist[str(idx)] = sum(self.distances[str(idx)]) / len(self.distances[str(idx)])
        else:
            self.avg_dist[str(idx)] = 0
        self.final_population = pop
        log.info(f'{idx}: InCluster: {self.in_cluster[str(idx)]} - Avg.Dists.: {self.avg_dist[str(idx)]} -'
                 f' Min Dist to Centroid: {str(min(self.centroid_distances[str(idx)]))[0:5]}'
                 f' Max Dist to Centroid: {str(max(self.centroid_distances[str(idx)]))[0:5]}')


    def population_metrics(self, pop, idx):
        import os, shutil
        prevIdx = list(self.distances.keys())[-1] if len(self.distances) > 0 else '0'
        self.distances[str(idx)] = []
        self.avg_dist[str(idx)] = []
        self.in_cluster[str(idx)] = 0
        self.in_cluster_percentage[str(idx)] = 0.0
        self.imgs[str(idx)] = []
        self.centroid_distances[str(idx)] = []

        outPath = os.path.join(self.problemPath, str(idx))
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        candidateList = []
        fitnessList = []
        final_population = []
        for candidate in pop:
            candidateList.append(candidate)
            fitnessList.append(candidate.F1)
        POPSIZE = 25 if len(fitnessList) > 25 else len(fitnessList)
        for i in range(0, POPSIZE):
            indx = fitnessList.index(max(fitnessList))
            final_population.append(candidateList[indx])
            fitnessList.remove(fitnessList[indx])
            candidateList.remove(candidateList[indx])
        pop = final_population
        in_cluster_nonduplicate = 0
        for ind in pop:
            self.centroid_distances[str(idx)].append(ind.F2)
            if ind.inCluster:
                if ind.export(outPath):
                    self.imgs[str(idx)].append(os.path.basename(ind.imgPath))
                    in_cluster_nonduplicate += 1
        shutil.rmtree(outPath)
        if prevIdx != '0' and (in_cluster_nonduplicate < self.in_cluster[prevIdx]):
            self.in_cluster[str(idx)] = self.in_cluster[prevIdx]
            self.imgs[str(idx)] = self.imgs[prevIdx]
            self.distances[str(idx)] = self.distances[prevIdx]
            self.in_cluster_percentage = self.in_cluster_percentage[prevIdx]
            self.avg_dist[str(idx)] = self.avg_dist[prevIdx]
            self.centroid_distances[str(idx)] = self.centroid_distances[prevIdx]
        else:
            for ind in pop:
                if ind.inCluster:
                    for ind2 in pop:
                        if ind2.inCluster:
                            param_dist = doParamDist(ind.params, ind2.params, self.config.upper_bound, self.config.lower_bound)
                            if param_dist != 0:
                                self.distances[str(idx)].append(param_dist)

            self.in_cluster[str(idx)] = in_cluster_nonduplicate
            self.in_cluster_percentage[str(idx)] = 100* (self.in_cluster[str(idx)]/POPSIZE)
            if len(self.distances[str(idx)]) > 0:
                self.avg_dist[str(idx)] = sum(self.distances[str(idx)])/len(self.distances[str(idx)])
            else:
                self.avg_dist[str(idx)] = 0
        if len(self.centroid_distances[str(idx)]) > 0:
            log.info(f'{idx}: InCluster: {self.in_cluster[str(idx)]} - Avg.Dists.: {self.avg_dist[str(idx)]} -'
                     f' Min Dist to Centroid: {str(min(self.centroid_distances[str(idx)]))[0:5]}'
                     f' Max Dist to Centroid: {str(max(self.centroid_distances[str(idx)]))[0:5]}')

