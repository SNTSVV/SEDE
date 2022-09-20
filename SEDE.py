import config
from imports import basename, argparse, os, shutil, join, np, exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')

    parser.add_argument('-a', '--action', help='supported actions: test, heatmap, cluster, assign, retrain',
                        required=False)
    parser.add_argument('-m', '--modelName', help='pretrainedWeights.pth Path', required=False)
    parser.add_argument('-o', '--outputPathX', help='Output path for saving the result', required=True)
    parser.add_argument('-sF', '--scratchFlag', help='Number of Classes', required=False)
    parser.add_argument('-n', '--ClusterModeX', help='ICD - WICD - S', required=False)
    parser.add_argument('-cF', '--clustF', help='clustering Flag', required=False)
    parser.add_argument('-dcF', '--drawCF', help='Exporting images Flag', required=False)
    parser.add_argument('-aF', '--assignF', help='Exporting images Flag', required=False)
    parser.add_argument('-daF', '--drawAssignF', help='Exporting images Flag', required=False)
    parser.add_argument('-err', '--errorMarginPixels', help='error Margin Pixels', required=False)
    parser.add_argument('-sub', '--faceSubSet', help='Subset of the face', required=False)
    parser.add_argument('-tl', '--transfer', help='scratch/pretrained', required=False)
    parser.add_argument('-rF', '--retrainF', help='HUDD, BL1, BL2', required=False)
    parser.add_argument('-mode', '--retrainMode', help='HUDD, BL1, BL2', required=False)
    parser.add_argument('-app', '--approach', help='A, B', required=False)
    parser.add_argument('-exp1', '--expNumber', help='Number of retrainings', required=False)
    parser.add_argument('-exp2', '--expNumber2', help='Number of retrainings', required=False)
    parser.add_argument('-ep', '--epoch', help='Number of epochs', required=False)
    parser.add_argument('-ass', '--assignMode', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-bs', '--BagSize', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-mc', '--maxClust', help='ICD - Centroid - Closest - SSE', required=False)
    parser.add_argument('-ow', '--ow', help='overwrite flag', required=False)
    parser.add_argument('-sel', '--select', help='layer selection mode', required=False)
    parser.add_argument('-fld', '--FLD', help='FLD selection mode', required=False)
    parser.add_argument('-wc', '--workersCount', help='FLD selection mode', required=False)
    parser.add_argument('-batchS', '--batchSize', help='FLD selection mode', required=False)
    parser.add_argument('-cleanF', '--cleanFlag', help='FLD selection mode', required=False)
    parser.add_argument('-rcc', '--rccSource', help='FLD selection mode', required=False)
    parser.add_argument('-numR', '--numRuns', help='FLD selection mode', required=False)
    parser.add_argument('-rA', '--retrieveAccuracy', help='FLD selection mode', required=False)
    parser.add_argument('-rq', '--RQ1A', help='FLD selection mode', required=False)
    parser.add_argument('-rS', '--retrainSet', help='FLD selection mode', required=False)
    parser.add_argument('-SEDE', '--SEDEmode', help='FLD selection mode', required=False)
    parser.add_argument('-iee', '--ieeVersion', default="1", help='iee_sim1, iee_sim2', required=False)
    parser.add_argument('-cls', '--clsNum', default="1", help='iee_sim1, iee_sim2', required=False)
    parser.add_argument('-case', '--caseStudy', default="FLD", help='FLD, HPD-H, HPD-F', required=False)
    args = parser.parse_args()


    #if args.SEDEmode == "RQ1":
    #    import SEDE_RQ1
    #    SEDE_RQ1.doRQ(args.caseStudy, os.getcwd())

    components = ["noseridge", "nose", "mouth", "rightbrow", "righteye", "lefteye", "leftbrow"]
    if args.ieeVersion is not None:
        if int(args.ieeVersion) == 1:
            config.nVar = 13
        elif int(args.ieeVersion) == 2:
            config.nVar = 23
        else:
            print("WARNING: number of Variables in config.nVar is not set")
    else:
        print("WARNING: number of Variables in config.nVar is not set")
    import Helper
    SEDE = Helper.Helper(outputPath=args.outputPathX, modelName=args.modelName, workersCount=args.workersCount,
                  batchSize=args.batchSize, metric="Euc", clustFlag=args.clustF, assignFlag=args.assignF,
                  retrainFlag=args.retrainF, retrainMode=args.retrainMode, retrainApproach=args.approach,
                  expNumber=args.expNumber, expNumber2=args.expNumber2, bagSize=args.BagSize,
                 clustMode=args.ClusterModeX, assMode=args.assignMode,
                  overWrite=args.ow, selectionMode=args.select, FLD=args.FLD, cleanFlag=args.cleanFlag,
                  RCC=args.rccSource, scratchFlag=args.scratchFlag, retrieveAccuracy=args.retrieveAccuracy,
                  RQ1A=False, retrainSet=args.retrainSet, drawClustFlag=args.drawCF, ieeVersion=args.ieeVersion, clustNum=args.clsNum)
    #HUDD.updateCaseFile()
    finalResultDict = {}
    datasetName = basename(args.outputPathX)
    TestSetCheck = False
    if args.SEDEmode == "HUDD":
        if datasetName == "FLD":
            if args.faceSubSet is None:
                maxSub = 0.0
                for subset in components:
                    print(subset)
                    # ResultDict, _ = HUDD.KPNet(subset)
                    # HUDD.faceSubset = subset
                    # HUDD.updateCaseFile()
                    # HUDD.saveResult()
            else:
                print(args.faceSubSet)
                ResultDict, _ = SEDE.KPNet(args.faceSubSet)
        else:
            ResultDict, _ = SEDE.AlexNet()

        if args.numRuns is None:
            if datasetName == "FLD":
                ResultDict, _ = SEDE.KPNet(components[0])
                SEDE.faceSubSet = components[0]
                # HUDD.saveResult()
            SEDE.retrainDNN()
        else:
            for x in range(0, int(args.numRuns)):
                SEDE.retrainDNN()
    elif args.SEDEmode == "SEDE":

        import searchModule, assignModule
        SEDE.RCC = "TR" #HPD-H
        #self.RCC = "TR1" #HPD-F
        SEDE.updateCaseFile()
        SEDE.selectLayer()
        print("Loading HM distance file for the selected layer.")
        HMDistFile = join(str(SEDE.caseFile["filesPath"]), str(SEDE.caseFile["selectedLayer"]) + "HMDistance.xlsx")
        clusterRadius, centroidHM, testHM = assignModule.getClusterData(SEDE.caseFile, HMDistFile)
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] #HPD-H
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #FLD/HPD-F
        clusters = [SEDE.clustNum]
        popSize = [25, 25, 25]
        nGen = [200, 100, 100]
        searchModule.search(SEDE.caseFile, clusters, centroidHM, clusterRadius, popSize, nGen)
    elif args.SEDEmode == "DeepJanus":
        from DeepJanus import main
        import assignModule
        SEDE.RCC = "TR" #HPD-H
        #self.RCC = "TR1" #HPD-F
        SEDE.updateCaseFile()
        SEDE.selectLayer()
        print("Loading HM distance file for the selected layer.")
        HMDistFile = join(str(SEDE.caseFile["filesPath"]), str(SEDE.caseFile["selectedLayer"]) + "HMDistance.xlsx")
        clusterRadius, centroidHM, testHM = assignModule.getClusterData(SEDE.caseFile, HMDistFile)
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] #HPD-H
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #FLD/HPD-F
        clusters = [SEDE.clustNum]
        main.runSEDE(SEDE.caseFile, clusters, centroidHM, clusterRadius)
    elif args.SEDEmode == "RQ2":
        import SEDE_RQ2, searchModule, assignModule
        SEDE.RCC = "TR" #HPD-H
        #self.RCC = "TR1" #HPD-F
        SEDE.updateCaseFile()
        SEDE.selectLayer()
        print("Loading HM distance file for the selected layer.")
        HMDistFile = join(str(SEDE.caseFile["filesPath"]), str(SEDE.caseFile["selectedLayer"]) + "HMDistance.xlsx")
        clusterRadius, centroidHM, testHM = assignModule.getClusterData(SEDE.caseFile, HMDistFile)
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] #HPD-H
        clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #FLD/HPD-F
        SEDE_RQ2.plotRQ(SEDE.caseFile, centroidHM)
    elif args.SEDEmode == "RQ3":
        import SEDE_RQ3, searchModule, assignModule
        SEDE.RCC = "TR" #HPD-H
        #self.RCC = "TR1" #HPD-F
        SEDE.updateCaseFile()
        SEDE.selectLayer()
        print("Loading HM distance file for the selected layer.")
        HMDistFile = join(str(SEDE.caseFile["filesPath"]), str(SEDE.caseFile["selectedLayer"]) + "HMDistance.xlsx")
        clusterRadius, centroidHM, testHM = assignModule.getClusterData(SEDE.caseFile, HMDistFile)
        #clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] #HPD-H
        clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #FLD/HPD-F
        SEDE_RQ3.plotRQ(SEDE.caseFile)
    elif args.SEDEmode == "RQ4":
        import SEDE_RQ4
        SEDE.RCC = "TR"
        SEDE.updateCaseFile()
        SEDE_RQ4.evaluateResults(SEDE.caseFile)
        import scipy.stats as stats
        import numpy as np
        while int(input("continue? 1: Yes, 2: No")) == 1:
            n = input("correctly classified images: ")
            total = input("total images: ")
            ob_table = np.array([[2800, 1210], [int(total), int(n)]])
            result = stats.chi2_contingency(ob_table, correction=False)  # correction = False due to df=1
            chisq, pvalue = result[:2]
            print('chisq = {}, pvalue = {}'.format(chisq, pvalue))
            result = stats.fisher_exact(ob_table)
            oddsr, pvalue = result[:2]
            print('fisher = {}, pvalue = {}'.format(oddsr, pvalue))
    elif args.SEDEmode == "RQ5":
        SEDE.selectLayer()
        SEDE.retrainDNN()
    elif args.SEDEmode == "testModel":
        SEDE.saveResult()
    elif args.SEDEmode == "generateHeatmaps":
        SEDE.generateHeatmaps()
    elif args.SEDEmode == "generateHMDists":
        SEDE.generateHMDistances()
    elif args.SEDEmode == "generateClusters":
        SEDE.generateClusters()
        SEDE.selectLayer()
    elif args.SEDEmode == "assignImages":
        SEDE.selectLayer()
        SEDE.assignImages()