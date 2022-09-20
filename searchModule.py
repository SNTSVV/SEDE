#
# Copyright (c) IEE, University of Luxembourg 2021-2022.
# Created by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2022.
#

import subprocess as sp
import pathlib as pl

from imports import np, random, makedirs, torch, pd, join, basename, exists, isfile, sys, os, cv2, dlib, dirname, \
    imageio, subprocess, shutil, math, random, distance, math, time, plt
from HeatmapModule import generateHeatMap, doDistance
from assignModule import getClusterData
from testModule import testModelForImg
#from dnnModels import ConvModel
#import ieedatavendor as ieeDV

import logging
# import mosa
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
#from pymoo.algorithms.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation, get_reference_directions
#from pymoo.visualization.scatter import Scatter
from pymoo.model.population import Population
from pymoo.model.individual import Individual
import nsga2P1
import nsga2plus
import pickle

import config as cfg

components = cfg.components
blenderPath = cfg.blenderPath
nVar = cfg.nVar
indvdSize = cfg.indvdSize
BL = cfg.BL
width = cfg.width
height = cfg.height
globalCounter = random.randint(1, 999999999)


def search(caseFile, clusters, centroidHM, clusterRadius, popSize, nGen):
    global layer
    global prevPOP
    layer = int(caseFile["selectedLayer"].replace("Layer", ""))
    outPath = join(caseFile["filesPath"], "Pool")
    caseFile["SimDataPath"] = outPath
    if not exists(outPath):
        makedirs(outPath)
    r = random.randint(1, 1000)
    print(r)
    GI_path = join(caseFile["filesPath"], "GeneratedImages")
    caseFile["GI"] = GI_path
    caseFile["iee_version"] = int(caseFile["iee_version"])
    if not exists(GI_path):
        makedirs(GI_path)
    dp = join(GI_path, "details" + str(r) + ".txt")
    open(dp, "w")
    for ID in clusters:
        file = open(dp, "a")
        file.write("Cluster:" + str(ID) + "\n")
        print("Cluster:" + str(ID) + "\n")
        logging.info("Searching images for 5Cluster:", ID)
        # evalImages(clsData, caseFile)
        caseFile["CR"] = clusterRadius[ID]
        caseFile["CH"] = centroidHM[ID]
        caseFile["ID"] = ID
        caseFile["GI"] = GI_path
        if not exists(join(GI_path, str(ID), "images")):
            makedirs(join(GI_path, str(ID), "images"))
        print("IEE Simulator", caseFile["iee_version"])
        if int(caseFile["iee_version"]) == 2:
            caseFile["xl"] = setX_2(1, "L")
            caseFile["xu"] = setX_2(1, "U")
        else:
            caseFile["xl"] = setX(1, "L")
            caseFile["xu"] = setX(1, "U")
        initpop = ent = None
        start1 = time.time()
        for i in range(0, len(popSize)):
            file.write("PopSize:" + str(popSize) + "\n")
            file.write("Iterations:" + str(nGen) + "\n")
            start = time.time()
            initpop, ent = doProblem(popSize[i], popSize[i], nGen[i], caseFile, initpop, ent, i+1)
            end = time.time()
            file.write(doTime("Problem " + str(i + 1) + " Cost:", start-end) + "\n")
        file.write(doTime("Total Cost:", start1) + "\n")
        file.close()



def doProblem(pop_size, popS_new, n_gen, caseFile, prev_pop, prev_en, probNum):
    dirPath = join(caseFile["GI"], str(caseFile["ID"]))
    if not exists(dirPath):
        makedirs(dirPath)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "PF" + str(probNum) + ".pop")
    init_ = [get_sampling("real_random"), prev_pop, prev_pop]
    out_ = ['w', 'w', 'a']
    noff_ = [1, 1, None]
    csv = [False, True, True]
    res = solveProblem(pop_size=pop_size, n_gen=n_gen, n_offsprings=noff_[probNum - 1], sampling=init_[probNum - 1],
                       caseFile=caseFile, probNum=probNum, initpop=prev_pop, ent=prev_en)
    # caseFile=caseFile, probNum=probNum - 1, initpop=prev_pop, ent=prev_en)
    if hasattr(res, "pop"):
        res = res.pop
    #PF = getPF(res.pop, CP)
    exportResults_2(res, "/PF/" + str(probNum) + ".txt", "/PF/" + str(probNum), probNum, dirPath,
                    # exportResults_2(res.pop, "/PF/" + str(probNum - 1) + ".txt", "/PF/" + str(probNum - 1), probNum - 1, dirPath,
                    open(dirPath + "/results.csv", out_[probNum - 1]), caseFile, prev_en, prev_pop, csv[probNum - 1])

    if probNum != 3:
        # new_pop, new_en = prepareINIT(res.pop, caseFile, dirPath, probNum + 1, popS_new)
        new_pop, new_en = prepareINIT(res, caseFile, dirPath, probNum, popS_new)
        #for member in new_pop:
        #    print(min(member.F))
        return new_pop, new_en
        #return None, None
    else:
        return None, None


def solveProblem(pop_size, n_gen, n_offsprings, sampling, caseFile, probNum, ent, initpop):
    problem = HUDDProblem_N(caseFile, probNum, ent, initpop, n_gen, pop_size)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "res" + str(probNum) + ".pop")
    CP2 = join(caseFile["GI"], str(caseFile["ID"]), "res" + str(probNum) + ".ckpt.npy")
    problem.CP = CP
    # if probNum == 0:
    #    algorithm = nsga2P0.NSGA2P0(pop_size=pop_size, sampling=sampling,
    #                                crossover=get_crossover("real_sbx", prob=0.7, eta=20),
    #                                mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    if probNum == 1:
        algorithm = nsga2P1.NSGA2P1(pop_size=pop_size, sampling=sampling,
                                    crossover=get_crossover("real_sbx", prob=0.7, eta=20),
                                    mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)

        #algorithm = NSGA2(pop_size=pop_size, sampling=sampling,
        #                            crossover=get_crossover("real_sbx", prob=0.7, eta=20),
        #                            mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    elif probNum == 2:
        algorithm = nsga2plus.NSGA2Plus(pop_size=pop_size, sampling=sampling,
                                        crossover=get_crossover("real_sbx", prob=0.3, eta=20),
                                        mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    # algorithm = NSGA3(ref_dirs=get_reference_directions("energy", pop_size, pop_size, seed=1), pop_size=pop_size,
    #                  sampling=sampling, crossover=get_crossover("real_sbx", prob=0.7, eta=20),
    #                  mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    elif probNum == 3:
        algorithm = nsga2plus.NSGA2Plus(pop_size=pop_size, sampling=sampling,
                                        crossover=get_crossover("real_sbx", prob=0.3, eta=20),
                                        mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
        # algorithm = NSGA3(ref_dirs=get_reference_directions("energy", pop_size, pop_size, seed=1), pop_size=pop_size,
        #                  sampling=sampling, crossover=get_crossover("real_sbx", prob=0.7, eta=20),
        #                  mutation=get_mutation("real_pm", prob=0.3), eliminate_duplicates=True)
    if isfile(CP):
        res = loadCP(CP)
        print("Loaded Checkpoint:", res)
    else:
        res = minimize(problem, algorithm, ('n_gen', n_gen), n_offsprings=n_offsprings, verbose=True)
        res = res.pop
        print(problem.dsim, problem.c)
        np.save(join(caseFile["GI"], str(caseFile["ID"]), "diversity.npy"), problem.dsim) #avg sim params distance
        #np.save(join(caseFile["GI"], str(caseFile["ID"]), "2.npy"), problem.dhm) #avg heatmap distances
        np.save(join(caseFile["GI"], str(caseFile["ID"]), "individuals.npy"), problem.c) #individuals belonging to cluster
        saveCP(res, CP)

    if hasattr(res, "pop"):
        res = res.pop
    for member in res:
        print(min(member.F))
    #exportResults_2(res, "/Pop/" + str(probNum) + ".txt", "/Pop/" + str(probNum), probNum,
    #                join(caseFile["GI"], str(caseFile["ID"])), None,
    #                caseFile, ent, initpop, False)
    return res


class HUDDProblem_N(Problem):
    def __init__(self, caseFile, probID, prevEN, prevPOP, n_gen, pop_size, **kwargs):
        self.outPath = caseFile["outputPath"]
        self.problemDict = {}
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.clusterRadius = caseFile["CR"]
        self.centroidHM = caseFile["CH"]
        self.caseFile = caseFile
        self.probID = probID
        self.prevEN = prevEN
        self.prevPOP = prevPOP
        self.previousPop = prevPOP
        self.t = time.time()
        self.dsim = []
        self.dhm = []
        self.dind = []
        self.c = []
        self.counter = 0
        dp = join(caseFile["GI"], str(caseFile["ID"]) + ".txt")
        open(dp, "w")
        self.file = open(dp, 'a')
        if probID == 1:
            #    n_obj = 2
            # elif probID == 0:
            n_obj = 1
        else:
            n_obj = pop_size
        if int(caseFile["iee_version"]) == 2:
            xl = setX_2(1, "L")
            xu = setX_2(1, "U")
        else:
            xl = setX(1, "L")
            xu = setX(1, "U")
        self.xl = xl
        self.xu = xu
        print("length of params", len(self.xl))
        print("configured length of params", nVar)
        assert len(self.xl) == nVar
        assert len(self.xu) == nVar
        self.file.write("Solving Problem .. " + str(probID) + "\n")
        logging.info("Number of Variables", nVar)
        logging.info("Number of Objectives", nVar)
        super().__init__(n_var=nVar, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=False, **kwargs,
                         type_var=np.float)

    def _evaluate(self, X, out, *args, **kwargs):  # Batchwise
        if not exists(self.outPath):
            makedirs(self.outPath)
        if self.probID == 1:
            # print("prevPop", self.previousPop)
            #F = HUDDevaluate_Pop1(X, self.caseFile, self.previousPop) #SEDE
            F = HUDDevaluate_Pop1N(X, self.caseFile, self.previousPop) #NSGA2
        elif self.probID == 2:
            F = HUDDevaluate_Pop2(X, self.caseFile, self.previousPop)
        elif self.probID == 3:
            F = HUDDevaluate_Pop3(X, self.caseFile, self.prevPOP)
        elif self.probID == 0:
            F = HUDDevaluate_Pop1(X, self.caseFile, self.previousPop)
        if out is None:
            return F
        else:
            out["F"] = np.array(F)
        self.file.write("Evaluation" + "\n")
        print("Offspring Evaluation:")
        #if len(F[0]) > 1:
        #    for n in F:
        #        self.file.write("F: " + str(min(n)) + "\n")
        #        print("F: " + str(min(n)))
        #else:
        #    self.file.write("F: " + str(F) + "\n")
        #    print("F:" + str(F))
        print(out["F"])
        d_sim = []
        d_hm = []
        d_ind = []
        if hasattr(self, "pop"): #NSGA
            self.previousPop = self.pop #NSGA
        #    print("has pop")
        #if self.previousPop is not None:
            # print("Comparing Current vs. Prev., F=1-2ndClosest")
        #    print("here")
            for x in self.previousPop:

                if int(self.caseFile["iee_version"]) == 2:
                    imgPath, F = generateHuman(x.X, self.caseFile)
                else:
                    imgPath, F = generateAnImage(x.X, self.caseFile)
                imgPath += ".png"

                if F:

                    N, DNNResult, P, L, D3, HM = doImage(imgPath, self.caseFile, self.caseFile["CH"])
                    print(D3 / self.caseFile["CR"])
                    if D3 /self.caseFile["CR"] <= 1:
                        d_ind.append(D3/self.caseFile["CR"])
                        for x3 in self.previousPop:
                            if int(self.caseFile["iee_version"]) == 2:
                                imgPath, F3 = generateHuman(x3.X, self.caseFile)
                            else:
                                imgPath, F3 = generateAnImage(x3.X, self.caseFile)
                            imgPath += ".png"
                            if F3:
                                N, DNNResult, P, L, D3, _ = doImage(imgPath, self.caseFile, HM[layer])
                                f3 = D3 / self.caseFile["CR"]
                                fp = doParamDist(x.X, x3.X, self.xl, self.xu)
                                if fp !=0 and f3 <= 1:
                                    d_sim.append(fp)
                                if D3 != 0:
                                    d_hm.append(D3)
        if len(d_sim) > 0:
            dsim_avg = sum(d_sim) / len(d_sim)
        else:
            dsim_avg = 0
        if len(d_hm)> 0:
            dhm_avg = sum(d_hm) / len(d_hm)
        else:
            dhm_avg = 0
        self.counter += 1
        self.dsim.append(dsim_avg)
        self.dhm.append(dhm_avg)
        self.c.append(self.counter)
        self.dind.append(100 * (len(d_ind)))

        print(d_ind, self.dind[-1], d_sim, d_hm, dsim_avg, dhm_avg,self.counter)

        self.file.write("Selected Population" + "\n")
        print("Selected Population:")
        if self.previousPop is not None:
            saveCP(self.previousPop, self.CP)
            for ind in self.previousPop:
                if len(ind.F) > 1:
                    self.file.write("F: " + str(min(ind.F)) + "\n")
                    print("F: " + str(min(ind.F)))
                else:
                    self.file.write("F: " + str(ind.F) + "\n")
                    print("F: " + str(ind.F))

    def _evaluate_EW(self, x, out, *args, **kwargs):  # Elementwise
        self.counter += 1
        print(str(100 * (self.counter / (self.n_gen * self.pop_size)))[0:5] + "%", end="\r")
        if not exists(self.outPath):
            makedirs(self.outPath)
        if self.probID == 1:
            problemDict = HUDDevaluate_N(x, self.caseFile, self.prevEN, self.prevPOP)
            self.problemDict[processX(x)] = problemDict
            out["F"] = np.array([problemDict["F1"]])
        elif self.probID == 0:
            problemDict = HUDDevaluate_N(x, self.caseFile, None, None)
            self.problemDict[processX(x)] = problemDict
            out["F"] = np.array([problemDict["F1"]])
        else:
            F_list = []
            for i in range(0, len(self.prevPOP)):
                problemDict = HUDDevaluate_N(x, self.caseFile, self.prevEN, self.prevPOP[i].X)
                self.problemDict[processX(x)] = problemDict
                if self.probID == 2:
                    F_list.append(problemDict["FY"])
                    out["F"] = np.array([problemDict["FY"]])
                elif self.probID == 3:
                    F_list.append(problemDict["FY_"])
                    out["F"] = np.array([problemDict["FY_"]])
            out["F"] = np.array(F_list)
        print(out["F"])
        # if time.time() - self.t > 2h:
        # terminate

#NSGA2
def HUDDevaluate_Pop1N(X, caseFile, Y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        if x is None:
            fy.append(math.inf)
            continue

        if int(caseFile["iee_version"]) == 2:
            imgPath, F = generateHuman(x, caseFile)
        else:
            imgPath, F = generateAnImage(x, caseFile)
        imgPath += ".png"
        dists = []
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
        else:
            f1 = math.inf
        f1_.append(f1)
        fy.append([fy_])
    # print("F1:", f1_)
    return f1_

#SEDE
def HUDDevaluate_Pop1(X, caseFile, Y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        if x is None:
            fy.append(math.inf)
            continue

        if int(caseFile["iee_version"]) == 2:
            imgPath, F = generateHuman(x, caseFile)
        else:
            imgPath, F = generateAnImage(x, caseFile)
        imgPath += ".png"
        dists = []
        fy_ = math.inf
        f1 = math.inf
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
            if f1 <= 1:
                # Compare current vs. previous population, get closest and 2nd closest
                if Y is not None:
                    # print("Comparing Current vs. Prev., F=1-2ndClosest")
                    for x2 in Y:
                        if int(caseFile["iee_version"]) == 2:
                            imgPath2, F2 = generateHuman(x2.X, caseFile)
                        else:
                            imgPath2, F2 = generateAnImage(x2.X, caseFile)
                        imgPath2 += ".png"
                        if x2.get("F")[0] <= 1:
                            N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                            _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])
                            fp = doParamDist(x, x2.X, xl, xu)
                            dists.append(fp)
                        else:
                            dists.append(math.inf)
                            # if x2.get("F")[0] <= 1:
                            #    dists.append(fp) #F=1-fp  0.7
                            # else: #face not found #image 2 outside cluster
                            # dists.append(math.inf) #F=1-inf (-inf)
                            # else
                    closestDist = min(dists)
                    dists.remove(closestDist)
                    closestDist = min(dists)
                    fy_ = 1 - closestDist
                else:
                    # print("Comparing Current vs. Current, F=1-2nClosest")
                    # print("Previous Population not found..")
                    # if previous population is None, closestDist is based on current population
                    dists_2 = []
                    for x3 in X:
                        if int(caseFile["iee_version"]) == 2:
                            imgPath, F3 = generateHuman(x3, caseFile)
                        else:
                            imgPath, F3 = generateAnImage(x3, caseFile)
                        imgPath += ".png"
                        if F3:
                            N, DNNResult, P, L, D3, _ = doImage(imgPath, caseFile, CH)
                            f3 = D3 / CR
                            fp = doParamDist(x, x3, xl, xu)
                            if f3 <= 1:
                                dists_2.append(fp)
                    if len(dists_2) > 1:
                        closestDist = min(dists_2)
                        dists_2.remove(closestDist)
                        closestDist = min(dists_2)
                        fy_ = 1 - closestDist
                    else:
                        fy_ = f1
            else:
                fy_ = f1
        f1_.append(f1)
        fy.append([fy_])
    # print("F1:", f1_)
    return fy


def HUDDevaluate_Pop2(X, caseFile, Y):
    print("POP2")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        l_D = []
        if int(caseFile["iee_version"]) == 2:
            for j in range(0, len(Y)):
                l_D.append(doParamDist(x, Y[j].X, setX_2(1, "L"), setX_2(1, "U")))
        else:
            for j in range(0, len(Y)):
                l_D.append(doParamDist(x, Y[j].X, setX(1, "L"), setX(1, "U")))
        closestIndex = np.argsort(np.array(l_D))[0]
        fyy = []
        fy_ = math.inf
        if x is None:
            for x2 in Y:
                fyy.append(fy_)
            continue
        if int(caseFile["iee_version"]) == 2:
            imgPath, F = generateHuman(x, caseFile)
        else:
            imgPath, F = generateAnImage(x, caseFile)
        imgPath += ".png"
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
            if f1 <= 1:
                j = 0
                for x2 in Y:
                    if int(caseFile["iee_version"]) == 2:
                        imgPath2, F2 = generateHuman(x2.X, caseFile)
                    else:
                        imgPath2, F2 = generateAnImage(x2.X, caseFile)
                    imgPath2 += ".png"
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])
                    fp = doParamDist(x, x2.X, xl, xu)
                    if j == closestIndex:
                        if not DNNResult:
                            fy_ = fp
                            fyy.append(fy_)
                            print("Eval-OK:", closestIndex, fy_)
                        else:
                            fy_ = 2 - N
                            fyy.append(fy_)
                            print("Eval-NO:", closestIndex, fy_)
                    else:
                        fy_ = 2 + fp
                        fyy.append(fy_)
                    j += 1
            else:
                for x2 in Y:
                    fy_ = 3 + f1
                    fyy.append(fy_)
        else:
            for x2 in Y:
                fyy.append(fy_)
        fy.append(fyy)
    return fy


def HUDDevaluate_Pop3(X, caseFile, Y):
    print("POP3")
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    fy = []
    f1_ = []
    for x in X:
        j = len(fy)
        fy_ = math.inf
        fyy = []
        l_D = []
        if caseFile["iee_version"] == 2:
            for j in range(0, len(Y)):
                l_D.append(doParamDist(x, Y[j].X, setX_2(1, "L"), setX_2(1, "U")))
        else:
            for j in range(0, len(Y)):
                l_D.append(doParamDist(x, Y[j].X, setX(1, "L"), setX(1, "U")))
        closestIndex = np.argsort(np.array(l_D))[0]
        if x is None:
            for x2 in Y:
                fyy.append(fy_)
            continue
        if caseFile["iee_version"] == 2:
            imgPath, F = generateHuman(x, caseFile)
        else:
            imgPath, F = generateAnImage(x, caseFile)
        imgPath += ".png"
        if F:
            N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)
            f1 = D / CR
            j = 0
            for x2 in Y:
                if caseFile["iee_version"] == 2:
                    imgPath2, F2 = generateHuman(x2.X, caseFile)
                else:
                    imgPath2, F2 = generateAnImage(x2.X, caseFile)
                imgPath2 += ".png"
                N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])
                fp = doParamDist(x, x2.X, xl, xu)
                if j == closestIndex:
                    if DNNResult:
                        fy_ = fp
                        fyy.append(fy_)
                        print("Eval-OK:", closestIndex, fy_)
                    else:
                        fy_ = 1 + N + abs(1 - f1)
                        fyy.append(fy_)
                        print("Eval-NO:", closestIndex, fy_)
                else:
                    fy_ = 2 + fp
                    fyy.append(fy_)
                j += 1
        else:
            for x2 in Y:
                fyy.append(math.inf)
        fy.append(fyy)
    return fy


def HUDDevaluate_N(x, caseFile, prevEN, y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    if caseFile["iee_version"] == 2:
        imgPath, F = generateHuman(x, caseFile)
    else:
        imgPath, F = generateAnImage(x, caseFile)
    imgPath += ".png"
    f2_ = None
    fy = math.inf
    fy_ = math.inf
    fp = None
    ND = None
    D2 = None
    if F:
        N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)

        f1 = D / CR
        fp = doParamDist(x, x, xl, xu)
        if DNNResult:
            AN = N
        else:
            AN = 1

        if f1 <= 1:
            f2 = 1 - AN
            f1_ = 1 - f1
        else:
            f2 = 1
            f1_ = f1

        if prevEN is not None:

            if not DNNResult:
                f2_ = 0
            else:
                f2_ = 1
        if y is not None:

            for y1 in y:
                if caseFile["iee_version"] == 2:
                    imgPath2, F2 = generateHuman(y1.X, caseFile)
                else:
                    imgPath2, F2 = generateAnImage(y1.X, caseFile)
                imgPath2 += ".png"
                if F2:
                    N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                    _, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])

                    ND = D2 / CR
                    fp = doParamDist(x, y1.X, xl, xu)
                    ND = fp
                    if math.isnan(fp):
                        ND = 0
                    if f1 < 1:
                        if not DNNResult:
                            fy = ND
                        else:
                            fy = 1 + (1 - N)
                    else:
                        fy = 2 + f1

                    if DNNResult:
                        fy_ = ND
                    else:
                        fy_ = 1 + N

                else:
                    print("face 2 not found")
    else:
        f1 = math.inf
        f1_ = math.inf
        f2 = math.inf
        f2_ = math.inf
        fy = math.inf
        fp = math.inf
        N = math.inf
        D = math.inf
        D2 = math.inf
        P = None
        L = None
        DNNResult = None
        AN = None
    problemDict = {"Face": F, "Prediction": P, "Label": L, "N_": N, "Adjusted_N": AN, "FY_": fy_,
                   "Medoid-Distance": D, "Y-Distance": D2, "DNNResult": DNNResult,
                   "F1": f1, "F1_": f1_, "F2": f2, "F2_": f2_, "FY": fy, "FP": fp, "ND": ND}
    return problemDict


def exportResults_2(PF, txt1, outPathPOP, popNum, dirPath, outFile, caseFile, prev_en, prev_pop, csvFlag):
    outPathPOP = dirPath + outPathPOP
    if not exists(outPathPOP):
        makedirs(outPathPOP)
    SimDataPath = caseFile["SimDataPath"]
    clusterImages = list()
    txt1 = dirPath + txt1
    file = open(txt1, "a")
    # if popNum == 1:
    #    F = HUDDevaluate_Pop1(PF.get("X"), caseFile, None)
    # if popNum == 2:
    #    F = HUDDevaluate_Pop2(PF.get("X"), caseFile, PF.get("X"))
    # if popNum == 3:
    #    F = HUDDevaluate_Pop3(PF.get("X"), caseFile, PF.get("X"))

    dub = 0
    for i in range(0, len(PF)):
        X = PF[i].X
        problemDict = HUDDevaluate(X, caseFile, None, None)
        if problemDict["Face"]:
            imgName = processX(X) + ".png"
            if not isfile(join(outPathPOP, imgName)):
                shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
            else:
                shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                dub += 1
            if problemDict["F1"] <= 1:
                if popNum == 0 or popNum == 1:
                        clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                        clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                if popNum == 2 and not problemDict["DNNResult"]:
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                if popNum == 3 and problemDict["DNNResult"]:
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                file.write(imgName + "\n")
                file.write(str(PF[i].F) + "\n")
                for nameX in problemDict:
                    file.write(str(nameX) + ": " + str(problemDict[nameX]) + "\n")
    if csvFlag:
        toCSV_N(PF, outFile, caseFile, prev_en, prev_pop, popNum)
    if len(clusterImages) > 1:
        imageio.mimsave(outPathPOP + "_" + str(len(clusterImages)) + '.gif', clusterImages)


def exportResults(PF, txt1, outPathPOP, popNum, dirPath, outFile, caseFile, prev_en, prev_pop, csvFlag):
    outPathPOP = dirPath + outPathPOP
    if not exists(outPathPOP):
        makedirs(outPathPOP)
    SimDataPath = caseFile["SimDataPath"]
    clusterImages = list()
    txt1 = dirPath + txt1
    file = open(txt1, "a")
    if popNum == 0 or popNum == 1 or popNum == 2 or popNum == 3:
        dub = 0
        for i in range(0, len(PF)):
            X = PF[i].X
            print(X)
            problemDict = HUDDevaluate(X, caseFile, None, None)
            if problemDict["Face"]:
                imgName = processX(X) + ".png"
                if not isfile(join(outPathPOP, imgName)):
                    shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
                else:
                    shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                    dub += 1
                clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                file.write(imgName + "\n")
                file.write(str(PF[i].F) + "\n")
                for nameX in problemDict:
                    file.write(str(nameX) + ": " + str(problemDict[nameX]) + "\n")
        if csvFlag:
            toCSV_N(PF, outFile, caseFile, prev_en, prev_pop, popNum)
        if len(clusterImages) > 1:
            imageio.mimsave(outPathPOP + "_" + str(len(clusterImages)) + '.gif', clusterImages)
    else:
        dub = 0
        for i in range(0, len(PF)):
            clusterImages = list()
            if prev_pop is not None:
                problemDict = HUDDevaluate2(PF[i].X, caseFile, prev_en, prev_pop[i])
            else:
                problemDict = HUDDevaluate2(PF[i].X, caseFile, prev_en, None)
            for j in range(0, indvdSize):
                X = getI(PF[i].X, j)
                if problemDict[processX(X)]["Face"]:
                    imgName = processX(X) + ".png"
                    clusterImages.append(imageio.imread(join(SimDataPath, imgName)))
                    if not isfile(join(outPathPOP, imgName)):
                        shutil.copy(join(SimDataPath, imgName), join(outPathPOP, imgName))
                    else:
                        shutil.copy(join(SimDataPath, imgName), join(outPathPOP, processX(X) + "_" + str(dub) + ".png"))
                        dub += 1
                    file.write(imgName + "\n")
                    file.write(str(PF[i].F) + "\n")
                    for nameX in problemDict[processX(X)]:
                        file.write(str(nameX) + ": " + str(problemDict[processX(X)][nameX]) + "\n")
            if len(clusterImages) > 1:
                imageio.mimsave(outPathPOP + str(i) + "_" + str(len(clusterImages)) + '.gif', clusterImages)
        if csvFlag:
            toCSV_N(PF[0], outFile, caseFile, prev_en, prev_pop[0], popNum)
    file.close()


def toCSV_N(pop, outFile, CF, prevEN, prev_pop, probNum):
    ID = CF["ID"]
    counter = 1
    problemDict = HUDDevaluate_N(pop[0].X, CF, prevEN, prev_pop)
    if probNum == 2:
        strW = "idx,clusterID"
        for _x_ in problemDict:
            if "Face" in problemDict:
                if problemDict["Face"]:
                    for nameX in problemDict:
                        if nameX != "FY":
                            strW += "," + str(nameX)
                    break
        #strW += ",cam_dir0,cam_dir1,cam_dir2,cam_loc0,cam_loc1,cam_loc2,lamp_loc0," \
        #        "lamp_loc1,lamp_loc2," \
        #        "head_pose0,head_pose1,head_pose2,pose,imgPath\r\n"
        strW += ",head0,head1,head2,lampcol0,lampcol1,lampcol2,lamploc0," \
                "lamploc1,lamploc2," \
                "lampdir0,lampdir1,lampdir2,cam,age,hue,iris,sat,val,freckle,oil,veins,eyecol,gender,imgPath\r\n"
        outFile.writelines(strW)
    if probNum == 3:
        ID = 0
    for j in range(0, len(pop)):

        problemDict = HUDDevaluate_N(pop[j].X, CF, prevEN, prev_pop)
        X = pop[j].X
        if problemDict["Face"]:
            if ((not problemDict["DNNResult"]) and (probNum == 2)) or ((problemDict["DNNResult"]) and (probNum == 3)):
                imgPath = join(CF["filesPath"], "Pool", processX(X) + ".png")
                strMerge = str(counter) + "," + str(ID)
                for nameX in problemDict:
                    if nameX != "FY":
                        strMerge += "," + str(problemDict[nameX])
                for j in range(0, len(X)):
                    strMerge += "," + str(X[j])
                #strMerge += "," + str(math.floor(X[len(X) - 1]))
                strMerge += "," + str(imgPath)
                strMerge += "\r\n"
                outFile.writelines(strMerge)
            counter += 1


def HUDDevaluate(x, caseFile, prevEN, y):
    xl = caseFile["xl"]
    xu = caseFile["xu"]
    CR = caseFile["CR"]
    CH = caseFile["CH"]
    imgPath, F = generateAnImage(x, caseFile)
    imgPath += ".png"
    f2_ = None
    fy = math.inf
    fp = None
    ND = None
    D2 = None
    HN = None
    LN = False
    if F:
        N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, CH)

        f1 = D / CR

        if DNNResult:
            AN = N
        else:
            AN = 1

        if f1 <= 1:
            f2 = 1 - AN
            f1_ = 1 - f1
        else:
            f2 = 1
            f1_ = f1

        if prevEN is not None:
            if AN >= max(prevEN):
                HN = True
            else:
                HN = False

            if (not DNNResult) or HN:
                f2_ = 0
            else:
                f2_ = 1

        if y is not None:

            if caseFile["iee_version"] == 2:
                imgPath2, F2 = generateHuman(y, caseFile)
            else:
                imgPath2, F2 = generateAnImage(y, caseFile)
            imgPath2 += ".png"
            if F2:
                N2, _, _, _, _, layersHM = doImage(imgPath2, caseFile, CH)
                N1, _, _, _, D2, _ = doImage(imgPath, caseFile, layersHM[layer])

                if N1 <= N2:
                    LN = True
                    HN = False
                ND = D2 / CR
                fp = doParamDist(x, y, xl, xu)
                ND = fp
                if DNNResult and LN:
                    fy = ND
                elif (not DNNResult) or (not LN):
                    if ND != 0:
                        # fy = 1 + (1 / f1)
                        fy = 1 + (N / f1)
                    else:
                        fy = math.inf
            else:
                print("face 2 not found")
    else:
        f1 = math.inf
        f1_ = math.inf
        f2 = math.inf
        f2_ = math.inf
        fy = math.inf
        fp = math.inf
        N = math.inf
        D = math.inf
        D2 = math.inf
        LN = False
        HN = True
        P = None
        L = None
        DNNResult = None
        AN = None
    problemDict = {"Face": F, "Prediction": P, "Label": L, "N_": N, "Adjusted_N": AN, "Low_N": LN, "High_N": HN,
                   "Medoid-Distance": D, "Y-Distance": D2, "DNNResult": DNNResult,
                   "F1": f1, "F1_": f1_, "F2": f2, "F2_": f2_, "FY_": fy, "FP": fp, "ND": ND}
    return problemDict


def compareX(x1, x2):
    equal = True
    for i in range(0, len(x1)):
        if x1[i] != x2[i]:
            equal = False
    return equal


def prepareINIT(PF, caseFile, dirPath, probNum, pop_size):
    print("Preparing Initial Population", probNum)
    CP = join(caseFile["GI"], str(caseFile["ID"]), "init" + str(probNum) + ".pop")
    N, _ = getEntropy(PF, probNum, caseFile)
    if isfile(CP):
        initpop = loadCP(CP)
        print("Loaded Initpop:", initpop)
    else:
        initpop = initPOP_Pop(PF, probNum, pop_size, N, caseFile)
        saveCP(initpop, CP)

    exportResults(initpop, "/Initial/" + str(probNum) + ".txt", "/Initial/" + str(probNum), probNum, dirPath,
                  None, caseFile, N, initpop, False)
    print("Initial population Size:", len(initpop))
    return initpop, N


def getEntropy(PF, probNum, caseFile):
    N = []
    PF2 = []
    for member2 in PF:
        problemDict = HUDDevaluate_N(member2.X, caseFile, None, None)
        if problemDict["N_"] is not None:
            N.append(problemDict["N_"])
    return N, PF2


def initPOP_N(PF, probNum, popLen, N, caseFile):
    pop2 = Population(popLen)
    for i in range(0, len(PF)):
        F_list = []
        if probNum == 2:
            key = "FY"
        elif probNum == 3:
            key = "FY_"
        for j in range(0, len(PF)):
            problemDict = HUDDevaluate_N(PF[i].X, caseFile, N, PF[j].X)
            F_list.append(problemDict[key])
        pop2[i] = Individual(X=PF[i].X, CV=PF[i].CV, feasible=PF[i].feasible, F=np.array(F_list))
    return pop2


def initPOP_Pop(PF, probNum, popLen, N, caseFile):
    pop2 = Population(popLen)
    X_list = []
    for ind in PF:
        X_list.append(ind.X)
    if probNum == 1:
        F = HUDDevaluate_Pop1(X_list, caseFile, PF)
    elif probNum == 2:
        F = HUDDevaluate_Pop2(X_list, caseFile, PF)
    else:
        F = HUDDevaluate_Pop3(X_list, caseFile, PF)
    for i in range(0, len(PF)):
        pop2[i] = Individual(X=PF[i].X, CV=PF[i].CV, feasible=PF[i].feasible, F=np.array(F[i]))
    return pop2


def getPF(pop1, CP):
    if isfile(CP):
        PF = loadCP(CP)
    else:
        print("Extracting Pareto Front")
        length = 0
        PF_ = []
        for i in range(0, len(pop1)):
            if pop1[i].get("rank") == 0:
                PF_.append(pop1[i])
                length += 1
        PF = Population(len(PF_))
        for i in range(0, len(PF)):
            PF[i] = PF_[i]
        print("Pareto Front #individuals:", length)
        saveCP(PF, CP)
    return PF


def cleanX(x, caseFile):
    goodIndex = []
    newX = []
    for j in range(0, indvdSize):
        _, face = generateAnImage(getI(x, j), caseFile)
        if face:
            goodIndex.append(j)
    for j in range(0, indvdSize):
        if j in goodIndex:
            for z in getI(x, j):
                newX.append(z)
        else:
            newGoodIndex = goodIndex[random.randint(0, len(goodIndex) - 1)]
            goodIndex.append(newGoodIndex)
            for z in getI(x, newGoodIndex):
                newX.append(z)
    return newX


def getParamVals():  # IEE_V1
    param_list = ["cam_look_0", "cam_look_1", "cam_look_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
                  "lamp_loc_0", "lamp_loc_1", "lamp_loc_2", "head_0", "head_1", "head_2"]
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
    return param_list, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU


def getParamVals_2():  # IEE_V2
    param_list = ["head0", "head1", "head2", "lampcol0", "lampcol1",
                  "lampcol2", "lamploc0", "lamploc1", "lamploc2", "lampdir0", "lampdir1", "lampdir2", "cam",
                  "age", "hue", "iris", "sat", "val", "freckle", "oil", "veins", "eyecol", "gender"]

    headL = [-50, -50, -50]
    headU = [50, 50, 50]
    lamp_colL = [0, 0, 0]
    lamp_colU = [1, 1, 1]
    lamp_locL = [0.361, -5.729, 16.54]
    lamp_locU = [0.381, -5.619, 16.64]
    lamp_dirL = [0.843, -0.85, 0.598]
    lamp_dirU = [0.893, -0.89, 0.798]
    camL = -0.5
    camU = 0.5
    ageL = -1
    hueL = irisL = satL = valL = freckleL = oilL = veinsL = eye_colL = genderL = 0
    hueU = irisU = satU = valU = freckleU = oilU = veinsU = ageU = 1
    genderU = 1
    eye_colU = 3  # math.floor(eye_colU)
    return param_list, headL, headU, lamp_colL, lamp_colU, lamp_locL, lamp_locU, lamp_dirL, lamp_dirU, camL, camU, \
           ageL, ageU, hueL, hueU, irisL, irisU, satL, satU, valL, valU, freckleL, freckleU, oilL, oilU, veinsL, \
           veinsU, eye_colL, eye_colU, genderL, genderU


def setX(size, ID):
    _, cam_dirL, cam_dirU, cam_locL, cam_locU, lamp_locL, lamp_locU, headL, headU, faceL, faceU = getParamVals()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in cam_dirL:
                xl.append(c)
            for c in cam_locL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
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
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            xl.append(random.uniform(faceL, faceU))
    return np.array(xl)


def setX_2(size, ID):
    _, headL, headU, lamp_colL, lamp_colU, lamp_locL, lamp_locU, lamp_dirL, lamp_dirU, camL, camU, \
    ageL, ageU, hueL, hueU, irisL, irisU, satL, satU, valL, valU, freckleL, freckleU, oilL, oilU, veinsL, \
    veinsU, eye_colL, eye_colU, genderL, genderU = getParamVals_2()
    xl = []
    if ID == "L":
        for i in range(0, size):
            for c in headL:
                xl.append(c)
            for c in lamp_colL:
                xl.append(c)
            for c in lamp_locL:
                xl.append(c)
            for c in lamp_dirL:
                xl.append(c)
            xl.append(camL)
            xl.append(ageL)
            xl.append(hueL)
            xl.append(irisL)
            xl.append(satL)
            xl.append(valL)
            xl.append(freckleL)
            xl.append(oilL)
            xl.append(veinsL)
            xl.append(eye_colL)
            xl.append(genderL)
    elif ID == "U":
        xl = []
        for i in range(0, size):
            for c in headU:
                xl.append(c)
            for c in lamp_colU:
                xl.append(c)
            for c in lamp_locU:
                xl.append(c)
            for c in lamp_dirU:
                xl.append(c)
            xl.append(camU)
            xl.append(ageU)
            xl.append(hueU)
            xl.append(irisU)
            xl.append(satU)
            xl.append(valU)
            xl.append(freckleU)
            xl.append(oilU)
            xl.append(veinsU)
            xl.append(eye_colU)
            xl.append(genderU)
    elif ID == "R":
        xl = []
        for i in range(0, size):
            for z in range(0, len(headU)):
                xl.append(random.uniform(headL[z], headU[z]))
            for z in range(0, len(lamp_colU)):
                xl.append(random.uniform(lamp_colL[z], lamp_colU[z]))
            for z in range(0, len(lamp_locU)):
                xl.append(random.uniform(lamp_locL[z], lamp_locU[z]))
            for z in range(0, len(lamp_dirU)):
                xl.append(random.uniform(lamp_dirL[z], lamp_dirU[z]))
            xl.append(random.uniform(camL, camU))
            xl.append(random.uniform(ageL, ageU))
            xl.append(random.uniform(hueL, hueU))
            xl.append(random.uniform(irisL, irisU))
            xl.append(random.uniform(satL, satU))
            xl.append(random.uniform(valL, valU))
            xl.append(random.uniform(freckleL, freckleU))
            xl.append(random.uniform(oilL, oilU))
            xl.append(random.uniform(veinsL, veinsU))
            xl.append(random.uniform(eye_colL, eye_colU))
            xl.append(random.uniform(genderL, genderU))
    return np.array(xl)


def getI(x, i):
    return x[nVar * i:nVar * (i + 1)]


def doParamDist(x, y, xl, xu):
    x_ = []
    y_ = []
    for m in range(0, len(x)):
        if xu[m] == xl[m]:
            x_.append(0)
            y_.append(0)
        else:
            x_.append((x[m] - xl[m]) / (xu[m] - xl[m]))
            y_.append((y[m] - xl[m]) / (xu[m] - xl[m]))
    d = distance.cosine(x_, y_)
    if math.isnan(d):
        return 0
    return d


def getParams(mini, maxi, param, BL):
    param_1st = param[0]
    param_3rd = param[1]
    if param_1st < mini:
        param_1st = mini
    if param_1st > maxi:
        param_1st = maxi
    if param_3rd < mini:
        param_3rd = mini
    if param_3rd > maxi:
        param_3rd = maxi
    if BL:
        return random.uniform(mini, maxi), random.uniform(mini, maxi)
    else:
        return param_1st, param_3rd


def getPosePath():
    model_folder = "mhx2"
    label_folder = "newlabel3d"
    pose_folder = "pose"
    model_file = ["Aac01_o", "Aaj01_o", "Aai01_c", "Aah01_o", "Aaf01_o", "Aag01_o", "Aab01_o", "Aaa01_o",
                  "Aad01_o"]  # TrainingSet
    # model_file = [model_folder + "/Aae01_o", model_folder + "/Aaa01_o"] #ImprovementSet1
    # model_file = [model_folder + "/Aae01_o"] #ImprovementSet2
    # model_file = ["Aad01_o"]  # TestSet
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aae01_o"] #TestSet1
    # model_file = [model_folder + "/Aad01_o", model_folder + "/Aah01_o"] #TestSet2
    label_file = ["aac01_o", "aaj01_o", "aai01_c", "aah01_o", "aaf01_o", "aag01_o", "aab01_o", "aaa01_o",
                  "aad01_o"]  # TrainingSet
    # label_file = [label_folder + "/aae01_o", label_folder + "/aaa01_o"] #ImprovementSet1
    # label_file = [label_folder + "/aae01_o"] #ImprovementSet2
    # label_file = ["aad01_o"]  # TestSet
    # label_file = [label_folder + "/aad01_o", label_folder + "/aae01_o"] #TestSet1
    # label_file = [label_folder + "/aad01_o", label_folder + "/aah01_o"] #TestSet2
    pose_file = ["P3", "P10", "P9", "P8", "P6", "P7", "P2", "P1", "P4"]  # TrainingSet
    # pose_file = [pose_folder + "/P5", pose_folder + "/P1"] #ImprovementSet1
    # pose_file = [pose_folder + "/P5"] #ImprovementSet2
    # pose_file = ["P4"]  # TestSet
    # pose_file = [pose_folder + "/P4", pose_folder + "/P5"] #TestSet1
    # pose_file = [pose_folder + "/P4", pose_folder + "/P8"] #TestSet2
    return model_folder, label_folder, pose_folder, model_file, label_file, pose_file


def generateHuman(x, caseFile):
    # x = [head0/1/2, lamp_col0/1/2, lamp_loc0/1/2, lamp_dir0/1/2, cam2, age, eye_hue, eye_iris, eye_sat,
    # eye_val, skin_freckle, skin_oil, skin_veins, eye_color, gender]
    # x[0:12] -- Scenario/Person
    # x[13:23] -- MBLab/Human
    global globalCounter
    #x[19] = math.floor(x[19])
    #print(x)
    SimDataPath = caseFile["SimDataPath"]
    imgPath = join(SimDataPath, str(globalCounter), processX(x))
    t1 = time.time()
    if not isfile(join(SimDataPath, processX(x)) + ".png"):
        # output = os.path.join(pl.Path(__file__).resolve().parent.parent.parent, 'snt_simulator', 'data', 'mblab')
        script = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V2", "mblab-interface", "scripts",
                              "snt_face_dataset_generation.py")
        data_path = join(pl.Path(__file__).parent.resolve(), "IEE_V2", "mblab_asset_data")
        cmd = [str(blenderPath), "-b", "--log-level", str(0), "-noaudio", "--python", script, "--", str(data_path), "-l", "debug", "-o",
               f"{imgPath}", "--render", "--studio-lights"]  # generate MBLab character
        try:
            devnull = open(join(SimDataPath, str(globalCounter) + "_MBLab_log.txt"), 'w')
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)
        except Exception as e:
            print("Error in MBLab creating")
            print(e)
            #exit(0)

        filePath = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V2", "ieekeypoints2.py")
        try:
            devnull = open(join(SimDataPath, str(globalCounter) + "_Blender_log.txt"), 'w')
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            cmd = [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
                 "--imgPath", str(imgPath)]
            #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
            sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)
            # str(imgPath)], stdout=subprocess.PIPE)
            shutil.copy(imgPath + ".png", join(SimDataPath, processX(x) + ".png"))
            shutil.copy(imgPath + ".npy", join(SimDataPath, processX(x) + ".npy"))
            shutil.rmtree(join(SimDataPath, str(globalCounter)))
        except Exception as e:
            print("error in Blender scenario creation")
            print(e)
            exit(0)
        # print(ls)
        globalCounter += 1
    img = cv2.imread(join(SimDataPath, processX(x)) + ".png")
    t2 = time.time()
    print("Image Generation: ", str(t2-t1)[0:5], end="\r")
    if img is None:
        print("image not found, not processed")
        #exit(0)
        return imgPath, False
    if len(img) > 128:
        faceFound = processImage(join(SimDataPath, processX(x)) + ".png", join(pl.Path(__file__).parent.resolve(),
                                                        "IEEPackage", "clsdata", "mmod_human_face_detector.dat"))
    else:
        faceFound = True

    #shutil.copy(join(SimDataPath, processX(x)) + ".png", join(caseFile["GI"], str(caseFile["ID"]), "images", processX(x) + ".png"))
    return join(SimDataPath, processX(x)), faceFound


def generateAnImage(x, caseFile):
    SimDataPath = caseFile["SimDataPath"]
    outPath = caseFile["outputPath"]
    model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
    m = random.randint(0, len(pose_file) - 1)
    m = int(math.floor(x[len(x) - 1]))
    imgPath = join(SimDataPath, processX(x))
    filePath = os.path.join(pl.Path(__file__).parent.resolve(), "IEE_V1", "ieekeypoints.py")
    # filePath = "./ieekeypoints.py"
    t1 = time.time()
    if not isfile(imgPath + ".png"):
        # ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
        #print(blenderPath)
        ls = subprocess.run(
            [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
             "--path",
             str(join(pl.Path(__file__).parent.resolve(), "IEEPackage")), "--model_folder",
             str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder), "--pose_file",
             str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
             #str(imgPath)])
             str(imgPath)], stdout=subprocess.PIPE)
             #str(imgPath)], capture_output=True, text=True, shell=True).stderr.strip("\n")
        print(ls)
        print(ls.stdout)
        # print(process.stderr)
    # process.wait()
    t2 = time.time()
    # print("Image Generation: ", str(t2-t1)[0:5], end="\r")
    # print(imgPath)
    img = cv2.imread(imgPath + ".png")
    if img is None:
        print("image not found, not processed")
        return imgPath, False
    if len(img) > 128:
        faceFound = processImage(imgPath + ".png", join(pl.Path(__file__).parent.resolve(), "IEEPackage", "clsdata",
                                                        "mmod_human_face_detector.dat"))
    else:
        faceFound = True
    # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
    # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
    #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
    #                                                   label_file[m])
    # print("Image processing", str(time.time()-t2)[0:5])
    return imgPath, faceFound


def doImage(imgPath, caseFile, centroidHM):
    layersHM, entropy = generateHeatMap(imgPath, caseFile["DNN"], caseFile["datasetName"], caseFile["outputPath"],
                                        False, None, None, caseFile["imgExt"], None)
    #lable = labelImage(imgPath)
    #DNNResult, pred = testModelForImg(caseFile["DNN"], lable, imgPath, caseFile)
    # if imgPath in Dist_Dict:
    if centroidHM is None:
        dist = 0
    else:
        dist = doDistance(centroidHM, layersHM[int(caseFile["selectedLayer"].replace("Layer", ""))], "Euc")
    # Dist_Dict[imgPath] = dist
    #return entropy, DNNResult, pred, lable, dist, layersHM
    return entropy, entropy, entropy, entropy, dist, layersHM


def doTime(strW, start):
    print(strW, math.ceil((time.time() - start) / 60.0), " mins")
    return str(strW + str(math.ceil((time.time() - start) / 60.0)) + " mins")


def saveCP(res, path):
    with open(path, 'wb') as config_dictionary_file:
        # Step 3
        pickle.dump(res, config_dictionary_file)


def loadCP(path):
    with open(path, 'rb') as config_dictionary_file:
        res = pickle.load(config_dictionary_file)
    return res


def labelBIWI(txtPath):
    with open(txtPath) as f:
        lines = f.readlines()
        HP = lines[4]
    HP1 = float(HP.split(" ")[0])
    HP2 = float(HP.split(" ")[1])

    if HP2 < 27:
        lab = "Top"
    elif 27 <= HP2 <= 66:
        lab = "Middle"
    else:
        lab = "Bottom"

    if HP1 < 42:
        lab += "Left"
    elif 42 <= HP1 <= 68:
        lab += "Center"
    else:
        lab += "Right"
    return lab


def processBIWI(imgPath, dlibPath, newimgPath):
    img = cv2.imread(imgPath)
    # img = img[200:600, 100:400]
    # cv2.imwrite(newimgPath, img)
    face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(faces) < 1:
        return False
    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h
    sw_0 = max(mx - 25 // 2, 0)
    sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
    sh_0 = max(my - 25 // 2, 0)
    sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
    assert sh_1 > sh_0
    assert sw_1 > sw_0
    big_face = new_img[sh_0:sh_1, sw_0:sw_1]
    new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
    x_data = new_img
    x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
    img = x_data
    cv2.imwrite(newimgPath, img)
    return True


def processImage(imgPath, dlibPath):
    img = cv2.imread(imgPath)
    npPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(npPath, allow_pickle=True)
    labelFile = configFile.item()['label']

    face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
    faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
    # img = new_img
    # labelFile = np.array(label_arr)
    width = img.shape[1]
    height = img.shape[0]
    mouth = [6, 7, 8, 23, 24, 25, 26]
    mouth = [32, 34, 36, 49, 52, 55, 58]

    for KP in mouth:
        x_p = labelFile[KP][0]
        y_p = img.shape[0] - labelFile[KP][1]
        px = x_p
        py = y_p
        if KP == 32:
            p1x = px
            p1y = py
        elif KP == 36:
            p2x = px
            p2y = py
        elif KP == 34:
            p7x = px
            p7y = py
        elif KP == 58:
            p3x = px
            p3y = py
        elif KP == 52:
            p4x = px
            p4y = py
        elif KP == 49:
            p5x = px
            p5y = py
        elif KP == 55:
            p6x = px
            p6y = py
    # new_img = putMask(imgPath, img, [p1x, p2x, p3x, p4x, p5x, p6x, p7x], [p1y, p2y, p3y, p4y, p5y, p6y, p7y])
    # cv2.imwrite(imgPath, new_img)
    # img = new_img
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return
    if len(faces) < 1:
        return False
    big_face = -np.inf
    mx, my, mw, mh = 0, 0, 0, 0
    for face in faces:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        if w * h > big_face:
            big_face = w * h
            mx, my, mw, mh = x, y, w, h
    sw_0 = max(mx - 25 // 2, 0)
    sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
    sh_0 = max(my - 25 // 2, 0)
    sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
    assert sh_1 > sh_0
    assert sw_1 > sw_0
    label_arr = []
    iee_labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70, 49, 52,
                  55, 58]
    for ky in iee_labels:
        if ky in labelFile:
            coord = [labelFile[ky][0], new_img.shape[0] - labelFile[ky][1]]
            label_arr.append(coord)  # (-1,-1) means the keypoint is invisible
        else:
            label_arr.append([0, 0])  # label does not exist
    new_label = np.zeros_like(np.array(label_arr))
    new_label[:, 0] = np.array(label_arr)[:, 0] - sw_0
    new_label[:, 1] = np.array(label_arr)[:, 1] - sh_0
    new_label[new_label < 0] = 0
    new_label[np.array(label_arr)[:, 0] == -1, 0] = -1
    new_label[np.array(label_arr)[:, 1] == -1, 1] = -1
    big_face = new_img[sh_0:sh_1, sw_0:sw_1]
    width_resc = float(128) / big_face.shape[0]
    height_resc = float(128) / big_face.shape[1]
    new_label2 = np.zeros_like(new_label)
    new_label2[:, 0] = new_label[:, 0] * width_resc
    new_label2[:, 1] = new_label[:, 1] * height_resc
    labelFile = new_label2
    new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
    #print(new_img.shape)
    x_data = new_img
    x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
    img = x_data
    cv2.imwrite(imgPath, img)
    return True


def labelImage(imgPath):
    margin1 = 10.0
    margin2 = -10.0
    margin3 = 10.0
    margin4 = -10.0
    configPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
    configFile = np.load(configPath, allow_pickle=True)
    configFile = configFile.item()
    HP1 = configFile['config']['head_pose'][0]
    HP2 = configFile['config']['head_pose'][1]
    originalDst = None
    if HP1 > margin1:
        if HP2 > margin3:
            originalDst = "BottomRight"
        elif HP2 < margin4:
            originalDst = "BottomLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "BottomCenter"
    elif HP1 < margin2:
        if HP2 > margin3:
            originalDst = "TopRight"
        elif HP2 < margin4:
            originalDst = "TopLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "TopCenter"
    elif margin2 <= HP1 <= margin1:
        if HP2 > margin3:
            originalDst = "MiddleRight"
        elif HP2 < margin4:
            originalDst = "MiddleLeft"
        elif margin4 <= HP2 <= margin3:
            originalDst = "MiddleCenter"
    if originalDst is None:
        print("cannot label img:", imgPath)
    return originalDst


def getANPD(pop, CF, paramFlag):
    xl = CF["xl"]
    xu = CF["xu"]
    CR = CF["CR"]
    global layer
    HMList = list()
    goodX = list()
    for i in range(0, len(pop)):
        imgPath, faceFound = generateAnImage(pop[i], CF)
        imgPath = imgPath + ".png"
        if faceFound:
            layersHMX, entropy = generateHeatMap(imgPath, CF["DNN"], CF["datasetName"],
                                                 CF["outputPath"],
                                                 False, None, None, CF["imgExt"], None)
            HMList.append(layersHMX[layer])
            goodX.append(pop[i])
    numPairs = int((len(HMList) * (len(HMList) - 1)) / 2)
    D = np.zeros(numPairs)
    k = 0
    if paramFlag:
        for i in range(0, len(goodX)):
            for j in range(0, len(goodX)):
                if j > i:
                    D[k] = doParamDist(goodX[i], goodX[j], xl, xu)
                    k += 1
    else:
        for i in range(0, len(HMList)):
            for j in range(0, len(HMList)):
                if j > i:
                    D[k] = doDistance(HMList[i], HMList[j], "Euc") / (2 * CR)
                    k = k + 1
    ANPD = np.average(D)
    if ANPD == 0:
        return -math.inf
    if math.isnan(ANPD):
        return math.inf
    else:
        return ANPD


def getF(problemDict, key):
    for any_ in problemDict:
        return np.array(problemDict[any_][key])


def concatX(PF):
    z = []
    j = random.randint(0, len(PF) - 1)
    while len(z) < nVar * indvdSize:
        b = 0
        while b < 3:
            for x1 in PF[j].X:
                z.append(x1)
            if len(z) == nVar * indvdSize:
                break
            b += 1
        if j < len(PF) - 1:
            j += 1
        else:
            j = 0
    assert len(z) == nVar * indvdSize
    return z


def processX(x):
    out = str(x[0])[0:5]
    for i in range(1, len(x)):
        out += "_" + str(x[i])[0:5]
    return out


def getClustersData(caseFile, heatMapDistanceExecl, clusterID):
    centroidRadius, centroidHMs = getClusterData(caseFile, heatMapDistanceExecl)
    return centroidRadius[clusterID], centroidHMs[clusterID]