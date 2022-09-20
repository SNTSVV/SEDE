import HeatmapModule
import retrainModule
from imports import torch, join, os, stat, setupTransformer, PathImageFolder, sc, pd, np, plt, random
import searchModule, assignModule


def generateO1(caseFile):
    layerClust = join(caseFile["filesPath"], "ClusterAnalysis_" + str(caseFile["clustMode"]),
                      caseFile["selectedLayer"] + ".pt")
    clsData = torch.load(layerClust, map_location=torch.device('cpu'))
    print("Loading HM distance file for the selected layer.")
    HMDistFile = join(caseFile["filesPath"], caseFile["selectedLayer"] + "HMDistance.xlsx")
    clusterRadius, centroidHM, testHM = assignModule.getClusterData(caseFile, HMDistFile)
    searchModule.search(caseFile, clsData, centroidHM, clusterRadius, [15], [30])
    return

def plotRQ(caseFile, centroidHM):
    layerClust = join(caseFile["filesPath"], "ClusterAnalysis_" + str(caseFile["clustMode"]),
                      caseFile["selectedLayer"] + ".pt")
    clsData = torch.load(layerClust, map_location=torch.device('cpu'))
    SEDE_imgs = getSEDE_imgs(clsData['clusters'], caseFile)
    RQ(centroidHM, SEDE_imgs, caseFile)
    return

def getSEDE_imgs(clusters, caseFile):
    print("IEE Simulator V", caseFile["iee_version"])
    dictVar = {}
    SEDE_imgs = {}
    GI = join(caseFile["filesPath"], "GeneratedImages")

    outPath = join(caseFile["filesPath"], "Pool")
    caseFile["SimDataPath"] = outPath
    if int(caseFile["iee_version"]) == 2:
        paramList, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = searchModule.getParamVals_2()
    else:
        paramList, _, _, _, _, _, _, _, _, _, _ = searchModule.getParamVals()
    for param in paramList:
        dictVar[param] = []
    for ID in clusters:

        SEDE_imgs[str(ID)] = []
        #path = join(GI, str(ID-7))
        path = join(GI, str(ID))
        #res = searchModule.loadCP(join(path, 'res1.pop'))
        imgs = []
        dirPath = join(path, 'Pop', '1')
        dir = os.listdir(join(path, 'Pop', '1'))
        res = searchModule.loadCP(join(path, "res1.pop"))
        if hasattr(res, "pop"):
            res = res.pop
        for ind in res:
            x = ind.X
            img = searchModule.processX(x)
            #imgPath = join(outPath, img + ".png") #fixme
            imgPath = join(dirPath, img + ".png") #fixme
            #print(imgPath)
            imgs.append(img.split(".png")[0])

            x = []
            imgParams = (img.split(".png")[0]).split("_")
            for j in imgParams:
                x.append(float(j))
            if len(x) == searchModule.nVar:
                SEDE_imgs[str(ID)].append(imgPath)

        i = 0
        for param in paramList:
            for img in imgs:
                x = []
                imgParams = (img.split(".png")[0]).split("_")
                #print(imgParams)
                for j in imgParams:
                    x.append(float(j))
                if len(x) > searchModule.nVar:
                    continue
                dictVar[param].append(x[i])
            i += 1
    print("Collected all O1 data")
    return SEDE_imgs

def RQ(centroidHM, SEDE_imgs, caseFile):
    print("SEDE Evaluation based on Heatmaps")
    layer = int(caseFile["selectedLayer"].replace("Layer", ""))
    dataTransformer = setupTransformer(caseFile["datasetName"])
    #data_dir = join(caseFile["DataSetsPath"], "TestSet_S_2.2k") #HPD-F
    data_dir = join(caseFile["DataSetsPath"], "TestSet_S_3k") #HPD-H
    transformedData = PathImageFolder(root=data_dir, transform=dataTransformer)
    caseFile["testDataNpy"] = caseFile["testDataSet"] = torch.utils.data.DataLoader(transformedData, batch_size=64, shuffle=True,
                                                   num_workers=4)
    o = caseFile["outputPath"]
    caseFile["outputPath"] = join(caseFile["filesPath"], "Heatmaps_S")

    retrainModule.alexTest(caseFile["modelPath"], caseFile["testDataSet"], None, caseFile["datasetName"], caseFile["DNN"], True,
                           join(caseFile["filesPath"], "simTestCSV.csv"))
    caseFile["testCSV"] = join(caseFile["filesPath"], "simTestCSV.csv")
    f = caseFile["filesPath"]
    caseFile["filesPath"] = join(caseFile["filesPath"], "Heatmaps_S")
    if not os.path.exists(caseFile["filesPath"]):
        os.makedirs(caseFile["filesPath"])
    HeatmapModule.saveHeatmaps(caseFile, "Test")
    HM_S, _ = HeatmapModule.collectHeatmaps_Dir(join(caseFile["outputPath"], "Heatmaps", caseFile["selectedLayer"])) #fixme
    cols = []
    vals = [] * len(HM_S)
    dists_unsafe_all = {}
    dists_SEDE_all = {}
    pval = {}
    nClusters = 0
    #clusters = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    #clusters = [1, 2, 4, 5, 6, 8, 10] #HPD-F
    clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #HPD-H
    for ID in clusters:
        #if ID == 4 or ID == 5 or ID == 6:
        #    IDx =  ID - 1
        #elif ID == 8:
        #    IDx = ID - 2
        #elif ID == 10:
        #    IDx = ID - 3
        #else:
        #    IDx = ID
        IDx = ID
        #cols.append("UI\n-"+str(IDx)) #HPD-F
        cols.append("UI\n-"+str(IDx+7)) #HPD-H
        #HM = centroidHM[ID-7]
        HM = centroidHM[ID]
        dists_unsafe = []
        #dists_all = []
        dists_SEDE = []
        #for img in os.listdir(data_dir): #todo: fix directory
        #    N, DNNResult, P, L, D3, layersHM = searchModule.doImage(join(data_dir, img), caseFile, HM)
        #    dists_all.append(HeatmapModule.doDistance(HM, layersHM[layer], "Euc"))
        for HM2 in HM_S:
            #print(HM2, HM_S[HM2])
            dists_unsafe.append(HeatmapModule.doDistance(HM, HM_S[HM2], "Euc"))
        #print(SEDE_imgs[str(ID)])
        for img2 in SEDE_imgs[str(ID)]:
            N, DNNResult, P, L, D3, layersHM = searchModule.doImage(img2, caseFile, HM)
            dists_SEDE.append(HeatmapModule.doDistance(HM, layersHM[layer], "Euc"))
        dists_unsafe_all[str(ID)] = dists_unsafe
        dists_SEDE_all[str(ID)] = dists_SEDE
        print("ID: ", ID)
        #print("Medoid-To-TestSet", sum(dists_all)/len(dists_all))
        print("Medoid-To-UnsafeTestSet", sum(dists_unsafe)/len(dists_unsafe))
        print("Medoid-To-SEDE", sum(dists_SEDE)/len(dists_SEDE))
        print("U:", sc.mannwhitneyu(dists_unsafe, dists_SEDE))
        pval[str(ID)] = sc.mannwhitneyu(dists_unsafe, dists_SEDE)
        #if ID == 4 or ID == 5 or ID == 6:
        #    IDx =  ID - 1
        #elif ID == 8:
        #    IDx = ID - 2
        #elif ID == 10:
        #    IDx = ID - 3
        #else:
        #    IDx = ID
        IDx = ID #HPD-H
        cols.append(str(IDx+7))
        #cols.append(str(IDx))
    #list1 = []
    #for i in range(0, len(dists_SEDE_all[str(8)])):
    for i in range(0, len(dists_SEDE_all[str(1)])):

        list1 = []
        for ID in clusters:
            list1.append(dists_unsafe_all[str(ID)][i])
            list1.append(dists_SEDE_all[str(ID)][i])
        vals.append(list1)

    df = pd.DataFrame(vals, columns=cols)
    #df = pd.DataFrame(vals, columns=['Original DNN\n(HPD-1)', 'Random-BL DNN\n(HPD-1)', 'HUDD DNN\n(HPD-1)',
    #                                 'SEDE DNN\n(HPD-1)'])
    ax = df.plot.box(grid='True')
    # ax.set_ylabel('Avg. Heatmaps Distance from RCC\'s medoid')
    # ax.set_ylabel('DNN Accuracy on Simulator Images (%)')
    ax.set_ylabel('Heatmap Distances from RCC\'s medoid')
    ax.set_xlabel('RCC (HPD-H)')
    figure = (ax).get_figure()
    # figure.savefig(join(GI_path, "RQ4.png"))
    figure.savefig(join(caseFile["outputPath"], "RQ2-HPDH.pdf"), bbox_inches = "tight")
    # figure.savefig(join(GI_path, "RQ4.pdf"))
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    return