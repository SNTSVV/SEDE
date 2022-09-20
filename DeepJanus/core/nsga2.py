import random
import time
import math
import pickle
import numpy
import os
from deap import base
from deap import creator
from deap import tools

from typing import List, Tuple
from DeepJanus.core.log_setup import get_logger
from DeepJanus.core.problem import Problem

log = get_logger(__file__)


def main(problem: Problem = None, seed=None, continueFlag=False, caseFile=None):


    config = problem.config
    creator.create("FitnessMulti", base.Fitness, weights=config.fitness_weights)
    creator.create("Individual", problem.deap_individual_class(), fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    # We need to define the individual, the evaluation function (OOBs), mutation
    # toolbox.register("individual", tools.initIterate, creator.Individual)
    toolbox.register("individual", problem.deap_generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", problem.deap_evaluate_individual)
    toolbox.register("mutate", problem.deap_mutate_individual)
    toolbox.register("select", tools.selNSGA2)
    if not continueFlag:
        t0 = time.time()
        random.seed(seed)

        # DEAP framework setup
        # We define a bi-objective fitness function.
        # 1. Maximize the sparseness minus an amount due to the distance between members
        # 2. Minimize the distance to the decision boundary


        #stats = tools.Statistics(lambda ind: ind.fitness.values)
        #stats.register("min", numpy.min, axis=0)
        #stats.register("max", numpy.max, axis=0)
        #stats.register("avg", numpy.mean, axis=0)
        #stats.register("std", numpy.std, axis=0)
        #logbook = tools.Logbook()
        #logbook.header = "gen", "evals", "min", "max", "avg", "std"

        # Generate initial population.
        log.info("### Initializing population....")
        pop = toolbox.population(n=config.POPSIZE)

        # Evaluate the initial population.
        # Note: the fitness functions are all invalid before the first iteration since they have not been evaluated.
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        #for ind in invalid_ind:
        #    toolbox.evaluate(ind)
        #problem.pre_evaluate_members(invalid_ind)

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        problem.archive.process_population(pop)

        # This is just to assign the crowding distance to the individuals (no actual selection is done).
        pop = toolbox.select(pop, len(pop))

        #record = stats.compile(pop)
        #logbook.record(gen=0, evals=len(invalid_ind), **record)
        #print(logbook.stream)

        # Initialize the archive.
        #problem.on_iteration(0, pop, logbook)

        # Begin the generational process
        condition = True
        total_gen_time = (time.time() - t0)/60
        gen = 1
    else:
        total_gen_time = config.total_gen_time
        gen = config.gen + 1
        pop = loadProblem(os.path.join(caseFile["loadPath"], "pop.obj"))
        problem.distances = loadProblem(os.path.join(caseFile["loadPath"], "distances.pkl"))
        problem.pop = pop
    log.info(f"total execution time: {total_gen_time}")
    while total_gen_time < config.GEN_MINUTES:

    #for gen in range(1, config.NUM_GENERATIONS):
        t1 = time.time()
        # invalid_ind = [ind for ind in pop]


        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #    ind.fitness.values = fit

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [ind.clone() for ind in offspring]
        log.info("reseed")
        problem.reseed(pop, offspring)

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        to_eval = offspring + pop
        invalid_ind = [ind for ind in to_eval]

        #problem.pre_evaluate_members(invalid_ind)

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        log.info("process archive")
        problem.archive.process_population(offspring + pop)
        # Select the next generation population
        log.info("selecting next generation population")
        pop = toolbox.select(pop + offspring, config.POPSIZE)
        t2 = time.time()
        gen_time = (t2 - t1)/60
        total_gen_time += gen_time
        #problem.population_metrics(pop, gen)
        #problem.population_metrics(pop, math.ceil(total_gen_time))
        problem.archive_metrics(pop, math.ceil(total_gen_time))
        problem.pop = pop
        config.total_gen_time = total_gen_time
        config.gen = gen
        log.info(f'N-GEN: {gen}')
        log.info(f'Gen Time (mins): {gen_time}')
        log.info(f'Total Exec. Time (mins): {total_gen_time}')
        log.info(f'Population Length: {len(pop)} individuals')
        gen += 1
        saveProblem(problem.pop, os.path.join(caseFile["loadPath"], "pop.obj"))
        saveProblem(problem.archive, os.path.join(caseFile["loadPath"], "archive.obj"))
        saveProblem(config, os.path.join(caseFile["loadPath"], "config.obj"))
        saveProblem(problem.distances, os.path.join(caseFile["loadPath"], "distances.pkl"))
        #record = stats.compile(pop)
        #logbook.record(gen=gen, evals=len(invalid_ind), **record)
        #print(logbook.stream)
        #problem.on_iteration(gen, pop, logbook)
    #return pop, logbook

    return pop, problem

def saveProblem(res, path):
    with open(path, 'wb') as config_dictionary_file:
        # Step 3
        pickle.dump(res, config_dictionary_file, pickle.HIGHEST_PROTOCOL)

def loadProblem(path):
    with open(path, 'rb') as config_dictionary_file:
        res = pickle.load(config_dictionary_file)
    return res
if __name__ == "__main__":
    #final_population, search_stats = main()
    final_population = main()
