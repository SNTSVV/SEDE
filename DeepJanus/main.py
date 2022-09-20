from os import makedirs
from os.path import join, exists
import pickle
import DeepJanus.core.nsga2
from DeepJanus.core.archive_impl import SmartArchive
from DeepJanus.core.log_setup import get_logger
#from self_driving.beamng_config import BeamNGConfig
import matplotlib.pyplot as plt
#import SEDE_config
import numpy as np
from DeepJanus import SEDE_config, problem, individual
import searchModule
log = get_logger(__file__)

#from SEDE_config import SEDEConfig
#from SEDE_problem import SEDEProblem
#from searchModule import setX, setX_2
#from self_driving.beamng_problem import BeamNGProblem

#config = BeamNGConfig()
#problem = BeamNGProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))
#problem = HPDProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))

def runSEDE(caseFile, clusters, centroidHM, clusterRadius):
    outPath = join(caseFile["filesPath"], "Pool")
    caseFile["SimDataPath"] = outPath
    if not exists(outPath):
        makedirs(outPath)
    caseFile["xl"] = searchModule.setX(1, "L")
    caseFile["xu"] = searchModule.setX(1, "U")
    caseFile["caseStudy"] = "FLD"
    if caseFile["datasetName"] == "HPD":
        caseFile["iee_version"] = int(caseFile["iee_version"])
        caseFile["caseStudy"] = "HPDF"
        if int(caseFile["iee_version"]) == 2:
            caseFile["xl"] = searchModule.setX_2(1, "L")
            caseFile["xu"] = searchModule.setX_2(1, "U")
            caseFile["caseStudy"] = "HPDH"
    #clusters = clusters[0:2]
    for i in range(caseFile["expNum1"], caseFile["expNum2"]):
        for clusterID in clusters:
            log.info(f'clusters: {clusters}')
            caseFile["CR"] = clusterRadius[clusterID]
            caseFile["cID"] = clusterID
            caseFile["runNum"] = str(i)
            problemPath = join(caseFile["filesPath"], "SEDE_DJ_Run" + str(i), "Cluster" + str(clusterID))
            configClass = SEDE_config.SEDEConfig(caseFile["xl"], caseFile["xu"], centroidHM[clusterID], caseFile)
            continueFlag = False
            if exists(join(problemPath, "pop.obj")):
                continueFlag = True
                log.info('loading problem')
                configClass = loadProblem(join(problemPath, "config.obj"))
                archive = loadProblem(join(problemPath, "archive.obj"))
                problemClass = problem.SEDEProblem(configClass, archive, continueFlag, caseFile)
                problemClass.archive = archive
                log.info(f'loaded probelm at generation time {configClass.total_gen_time}')
            else:
                log.info(f"initializing problem with popsize = {configClass.POPSIZE} and time limit = "
                         f"{configClass.GEN_MINUTES} minutes")
                problemClass = problem.SEDEProblem(configClass, SmartArchive(configClass.ARCHIVE_THRESHOLD),
                                                   continueFlag, caseFile)
            problemClass.archive.lower_bound = caseFile["xl"]
            problemClass.archive.upper_bound = caseFile["xu"]
            problemClass.archive.ARCHIVE_THRESHOLD = 0.0
            problemClass.problemPath = problemPath
            final_population, problemClass = DeepJanus.core.nsga2.main(problemClass, None, continueFlag,
                                                                       join(problemPath, "pop.obj"))
            final_population = problemClass.final_population
            log.info(f'Cluster: {clusterID}')
            log.info(f'avg distances {problemClass.avg_dist}')
            log.info(f'final population length {len(final_population)}')
            outPath = join(caseFile["filesPath"], "SEDE_DJ_Run" + str(i),"Cluster"+str(clusterID))
            if not exists(outPath):
                makedirs(outPath)
            in_cluster_nonduplicate = 0
            for ind in final_population:
                if ind.inCluster:
                    ind.export(outPath)
            #problem.in_cluster[str(config.GEN_MINUTES)] = in_cluster_nonduplicate
            log.info(f'in cluster: {problemClass.in_cluster}')
            log.info(f'in cluster %: {problemClass.in_cluster_percentage} %')
            np.save(join(outPath, "diversity.npy"), problemClass.avg_dist)
            np.save(join(outPath, "individuals.npy"), problemClass.in_cluster)
            np.save(join(outPath, "clusters.npy"), problemClass.in_cluster_percentage)
            log.info('done')

            plt.ioff()
            plt.show()

def loadProblem(path):
    with open(path, 'rb') as config_dictionary_file:
        res = pickle.load(config_dictionary_file)
    return res