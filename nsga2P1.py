#
# Copyright (c) IEE, University of Luxembourg 2021-2022.
# Created by Fabrizio Pastore, fabrizio.pastore@uni.lu, SNT, 2022.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2022.
#

from imports import np, distance, math, random
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm

from pymoo.model.evaluator import Evaluator
from pymoo.model.algorithm import Algorithm
from pymoo.docs import parse_doc_string
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.dominator import Dominator
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.model.population import Population


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    if P.shape[1] != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(P.shape[0], np.nan)
    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible
        else:
            if tournament_type == 'comp_by_dom_and_crowding':
                S[i] = compare(a, pop[a].get("F"), b, pop[b].get("F"),
                               method='smaller_is_better', return_random_if_equal=True)
            else:
                raise Exception("Unknown tournament type.")
    return S[:, None].astype(int, copy=False)


# =========================================================================================================
# Implementation
# =========================================================================================================

class NSGA2P1(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SimulatedBinaryCrossover(eta=15, prob=0.9),
                 mutation=PolynomialMutation(prob=None, eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=MultiObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=RankAndCrowdingSurvival(),
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.survival = False

    def _initialize(self): #initialize 5 populations

        if self.problem.probID == 1:
            print("Initalize")
            # create the initial population
            self.temp = self.initialization.do(self.problem, self.pop_size, algorithm=self)
            self.temp.set("n_gen", self.n_gen)

            # then evaluate using the objective function
            self.evaluator.eval(self.problem, self.temp, algorithm=self)

            # then evaluate using the objective function
            self.evaluator.eval(self.problem, self.temp, algorithm=self)
            if len(self.temp.get("F")[0]) > 1:
                mini = min(self.temp.get("F"))[0]
            else:
                mini = min(self.temp.get("F"))
            s = 1
            j = 0
            list_Temp = [self.temp]
            min_F = [mini]
            while(min(list_Temp[min_F.index(min(min_F))].get("F"))[0] > s):
                print(mini)
                self.temp = self.initialization.do(self.problem, self.pop_size, algorithm=self)
                self.temp.set("n_gen", self.n_gen)
                self.evaluator.eval(self.problem, self.temp, algorithm=self)

                list_Temp.append(self.temp)
                if len(self.temp.get("F")[0]) > 1:
                    mini = min(self.temp.get("F"))[0]
                else:
                    mini = min(self.temp.get("F"))
                min_F.append(mini)
                j += 1
                if j > 4:
                    #s += 0.5
                    break
            print("min_F", min(min_F))
            pop = self.problem.previousPop = list_Temp[min_F.index(min(min_F))]
            # that call is a dummy survival to set attributes that are necessary for the mating selection
            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                       n_min_infeas_survive=self.min_infeas_pop_size)
            self.pop, self.off = pop, pop
            print("initalized")
        else:

            self.temp = self.initialization.do(self.problem, self.pop_size, algorithm=self)
            self.temp.set("n_gen", self.n_gen)
            self.evaluator.eval(self.problem, self.temp, algorithm=self)
            pop = self.temp
            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                       n_min_infeas_survive=self.min_infeas_pop_size)
            self.pop, self.off = pop, pop

    def _initialize_random(self):
            self.temp = self.initialization.do(self.problem, self.pop_size, algorithm=self)
            self.temp.set("n_gen", self.n_gen)
            self.evaluator.eval(self.problem, self.temp, algorithm=self)
            pop = self.temp
            if self.survival:
                pop = self.survival.do(self.problem, pop, len(pop), algorithm=self,
                                       n_min_infeas_survive=self.min_infeas_pop_size)
            self.pop, self.off = pop, pop
    def _set_optimum(self, **kwargs):
        #print("Set optimum")
        self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        #self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        #print(self.opt.get("F"))

        #if not has_feasible(self.pop):
        #    self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        #else:
        #    self.opt = self.pop[self.pop.get("rank") == 0]

    def _next(self):
        #for i in range(0, len(self.pop)):

        # do the mating using the current population
        self._set_optimum()
        self.off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        print("Offspring generated..")
        self.off.set("n_gen", self.n_gen)

        if len(self.off) == 0:
            self.termination.force_termination = True
            return
        elif len(self.off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")
        continueReplace = True
        while continueReplace:
            lenOff = len(self.off)
            print("Evaluating offsprings")
            self.evaluator.eval(self.problem, self.off) #FIXME very expensive
            self.off = setClosest(self.off, self.pop)
            _index_off = sortF(self.off)
            _index_pop = sortF(self.pop)
            for i in _index_off:
                if not checkF(self.pop):
                    j = _index_pop[len(_index_pop) - 1]
                    if self.off[i].get("F")[0] < self.pop[j].get("F")[0]:
                        s = self.pop.get("X").tolist()
                        if not (self.off[i].X).tolist() in s:
                            self.pop[j] = self.off[i].copy(deep=True)
                            self.pop[j].F = self.off[i].F
                            self.off[i] = None
                            lenOff -= 1
                        #print("Optimizing Out-Space")
                else:
                    if self.off[i].get("F")[0] <= 1:
                        toReplace = self.off[i].get("closest")
                        if self.off[i].get("F")[0] < self.pop[toReplace].get("F")[0]:
                            self.pop[toReplace] = self.off[i].copy(deep=True)
                            self.pop[toReplace].F = self.off[i].F
                            self.off[i] = None # empty Individual
                            lenOff -= 1
                            #print("Diversifying In-Space")
                    #else:
                        #j = _index_pop[len(_index_pop)-1]
                        #if self.pop[j].get("F")[0] > 1:
                        #    if self.off[i].get("F")[0] < self.pop[j].get("F")[0]:
                        #        self.pop[j] = self.off[i].copy(deep=True)
                        #        self.pop[j].F = self.off[i].F
                        #        self.off[i] = None
                        #        lenOff -= 1
                        #        print("Optimizing Out-Space")
            if len(self.off) == lenOff:
                continueReplace = False
            #print("Removed", len(self.off) - lenOff, "offspring elements")
            self.off = updateOff(self.off, lenOff)
            #print("Updating the best population")
            self.problem.previousPop = None
            F_update = self.problem._evaluate(self.pop.get("X"), None)
            for i in range(0, len(self.pop)):
                self.pop[i].F = F_update[i]
            self.problem.previousPop = self.pop

            self._set_optimum()
        F_update.sort()
        print("FY:", F_update)




# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------
def is_sublist(a, b):
    if not a: return True
    if not b: return False
    return b[:len(a)] == a or is_sublist(a, b[1:])
def checkF(pop):
    for i in range(0, len(pop)):
        if pop[i].get("F")[0] > 1:
            return False
    return True
def updateOff(off, lenOff):
    new = Population(lenOff)
    i = 0
    for indv in off:
        if indv is not None:
            new[i] = indv
            i += 1
    return new
def setClosest(off, pop):
    for i in range(0, len(off)):
        l_D = []
        for j in range(0, len(pop)):
            l_D.append(doParamDist(off[i].X, pop[j].X, setX(1, "L"), setX(1, "U")))
        off[i].set("closest", np.argsort(np.array(l_D))[0])
    return off
def sortF(off):
    l_F_off = []
    for n in off.get("F"):
        l_F_off.append(n[0])
    _F = off.get("F")
    return np.argsort(np.array(l_F_off))

class RankAndCrowdingSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()

    def _do(self, problem, pop, n_survive, D=None, **kwargs):
        #added by fabrizio
        #self.evaluator.eval(self.problem, pop, algorithm=self)

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []
        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-24)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding


parse_doc_string(NSGA2P1.__init__)


def doParamDist(x, y, xl, xu):
    x_ = []
    y_ = []
    for m in range(0, len(x)):
        if (not math.isnan(x[m])):
            x_.append((x[m] - xl[m]) / (xu[m] - xl[m]))
        else:
            x_.append(0)
        if (not math.isnan(y[m])):
            y_.append((y[m] - xl[m]) / (xu[m] - xl[m]))
        else:
            y_.append(0)
    d = distance.cosine(x_, y_)

    return d


def setX(size, ID):
    _, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, lamp_colL, lamp_colU, lamp_dirL, \
    lamp_dirU, lamp_engL, lamp_engU, headL, headU, faceL, faceU = getParamVals()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in cam_dirL:
                xl.append(c)
            for c in cam_locL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
            for c in lamp_colL:
                xl.append(c)
            for c in lamp_dirL:
                xl.append(c)
            xl.append(lamp_engL)
            for c in headL:
                xl.append(c)
            xl.append(faceL)
    elif ID == "U":
        xl = []
        for i in range(0, size):
            for c in cam_dirU:
                xl.append(c)
            for c in cam_locU:
                xl.append(c)
            for c in lamp_locU:
                xl.append(c)
            for c in lamp_colU:
                xl.append(c)
            for c in lamp_dirU:
                xl.append(c)
            xl.append(lamp_engU)
            for c in headU:
                xl.append(c)
            xl.append(faceU)
    elif ID == "R":
        xl = []
        for i in range(0, size):
            for z in range(0, len(cam_dirU)):
                xl.append(random.uniform(cam_dirL[z], cam_dirU[z]))
            for z in range(0, len(cam_locU)):
                xl.append(random.uniform(cam_locL[z], cam_locU[z]))
            for z in range(0, len(lamp_locU)):
                xl.append(random.uniform(lamp_locL[z], lamp_locU[z]))
            for z in range(0, len(lamp_colU)):
                xl.append(random.uniform(lamp_colL[z], lamp_colU[z]))
            for z in range(0, len(lamp_dirU)):
                xl.append(random.uniform(lamp_dirL[z], lamp_dirU[z]))
            xl.append(random.uniform(lamp_engL, lamp_engU))
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            xl.append(random.uniform(faceL, faceU))
    return np.array(xl)




def getParamVals():
    param_list = ["cam_look_0", "cam_look_1", "cam_look_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
                  "lamp_loc_0", "lamp_loc_1", "lamp_loc_2", "lamp_direct_0", "lamp_direct_1", "lamp_direct_2",
                  "lamp_color_0", "lamp_color_1", "lamp_color_2", "head_0", "head_1", "head_2", "lamp_energy"]
    # TrainingSet parameters (min - max)
    cam_dirL = [-0.10, -4.67, -1.69]
    # cam_dirL = [-0.08, -4.29, -1.27] # constant
    cam_dirU = [-0.08, -4.29, -1.27]
    cam_locL = [0.261, -5.351, 14.445]
    # cam_locL = [0.293, -5.00, 14.869] # constant
    cam_locU = [0.293, -5.00, 14.869]  # constant
    lamp_locL = [0.361, -5.729, 16.54]
    # lamp_locL = [0.381, -5.619, 16.64] # constant
    lamp_locU = [0.381, -5.619, 16.64]  # constant
    lamp_colL = [1.0, 1.0, 1.0]  # constant
    lamp_colU = [1.0, 1.0, 1.0]  # constant
    lamp_dirL = [0.873, -0.87, 0.698]  # constant
    lamp_dirU = [0.873, -0.87, 0.698]  # constant
    lamp_engL = 1.0  # constant
    lamp_engU = 1.0  # constant
    headL = [-41.86, -79.86, -64.30]
    headU = [36.87, 75.13, 51.77]
    faceL = 0
    faceU = 8
    # TestSet parameters (min - max)
    # headL = [-32.94, -88.10, -28.53]
    # headU = [33.50, 74.17, 46.17]
    # fixing HP_2
    # headL = [-32.94, -88.10, -0.000001]
    # headU = [33.50, 74.17, 0]
    return param_list, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, lamp_colL, lamp_colU, lamp_dirL, \
           lamp_dirU, lamp_engL, lamp_engU, headL, headU, faceL, faceU
