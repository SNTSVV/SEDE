import HeatmapModule
import retrainModule
from imports import torch, join, os, stat, setupTransformer, PathImageFolder, sc, pd, np, plt, random, exists
import searchModule, assignModule

def getSEDE_imgs(clusters, caseFile):
    print("IEE Simulator V", caseFile["iee_version"])
    dictVar = {}
    SEDE_imgs = {}
    GI = join(os.getcwd(), "RQ2-3", caseFile["caseStudy"])
    #GI = join(caseFile["filesPath"], "GeneratedImages")

    outPath = join(caseFile["filesPath"], "Pool")
    caseFile["SimDataPath"] = outPath
    if int(caseFile["iee_version"]) == 2:
        paramList, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = searchModule.getParamVals_2()
    else:
        paramList, _, _, _, _, _, _, _, _, _, _ = searchModule.getParamVals()
    for param in paramList:
        dictVar[param] = []
    for ID in clusters:
        path = join(GI, "RCC-"+str(ID))
        if not exists(path):
            continue
        SEDE_imgs[str(ID)] = []
        #path = join(GI, str(ID-7))
        imgs = []
        #dirPath = join(path, 'Pop', '1')
        #dir = os.listdir(join(path, 'Pop', '1'))
        dir = os.listdir(path)
        #res = searchModule.loadCP(join(path, "res1.pop"))
        #if hasattr(res, "pop"):
        #    res = res.pop
        for ind in dir:
            #x = ind.X
            #img = searchModule.processX(x)
            #imgPath = join(outPath, img + ".png") #fixme
            #imgPath = join(dirPath, img + ".png") #fixme
            if ind.endswith(".png"):
                imgs.append(ind.split(".png")[0])
                x = []
                imgParams = (ind.split(".png")[0]).split("_")
                for j in imgParams:
                    x.append(float(j))
                if len(x) == searchModule.nVar:
                    SEDE_imgs[str(ID)].append(join(path, ind))

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
    #dataTransformer = setupTransformer(caseFile["datasetName"])
    #data_dir = join(caseFile["DataSetsPath"], "TestSet_S_2.2k") #HPD-F
    #data_dir = join(caseFile["DataSetsPath"], "TestSet_S_3k") #HPD-H
    #transformedData = PathImageFolder(root=data_dir, transform=dataTransformer)
    #caseFile["testDataNpy"] = caseFile["testDataSet"] = torch.utils.data.DataLoader(transformedData, batch_size=64, shuffle=True,
    #                                               num_workers=4)
    #o = caseFile["outputPath"]
    #caseFile["outputPath"] = join(caseFile["filesPath"], "Heatmaps_S")

    #retrainModule.alexTest(caseFile["modelPath"], caseFile["testDataSet"], None, caseFile["datasetName"], caseFile["DNN"], True,
    #                       join(caseFile["filesPath"], "simTestCSV.csv"))
    #caseFile["testCSV"] = join(caseFile["filesPath"], "simTestCSV.csv")
    #f = caseFile["filesPath"]
    #caseFile["filesPath"] = join(caseFile["filesPath"], "Heatmaps_S")
    #if not os.path.exists(caseFile["filesPath"]):
    #    os.makedirs(caseFile["filesPath"])
    #HeatmapModule.saveHeatmaps(caseFile, "Test")
    HM_S, _ = HeatmapModule.collectHeatmaps_Dir(join(caseFile["filesPath"], "Heatmaps_S", "Heatmaps", caseFile["selectedLayer"])) #fixme
    cols = []
    vals = [] * len(HM_S)
    colors = []
    dists_unsafe_all = {}
    dists_SEDE_all = {}
    pval = {}
    IDx = 1
    for ID in SEDE_imgs:
        #cols.append("UI\n-"+str(IDx))
        cols.append("UI-"+str(IDx))
        colors.append("red")
        HM = centroidHM[int(ID)]
        dists_unsafe = []
        dists_SEDE = []
        for HM2 in HM_S:
            dists_unsafe.append(HeatmapModule.doDistance(HM, HM_S[HM2], "Euc"))
        for img2 in SEDE_imgs[str(ID)]:
            N, DNNResult, P, L, D3, layersHM = searchModule.doImage(img2, caseFile, HM)
            dists_SEDE.append(HeatmapModule.doDistance(HM, layersHM[layer], "Euc"))
        dists_unsafe_all[str(ID)] = dists_unsafe
        dists_SEDE_all[str(ID)] = dists_SEDE
        print("ID: ", ID)
        print("Medoid-To-UnsafeTestSet", sum(dists_unsafe)/len(dists_unsafe))
        print("Medoid-To-SEDE", sum(dists_SEDE)/len(dists_SEDE))
        print("U:", sc.mannwhitneyu(dists_unsafe, dists_SEDE))
        pval[str(ID)] = sc.mannwhitneyu(dists_unsafe, dists_SEDE)
        cols.append(str(IDx))
        colors.append("blue")
        IDx += 1
    for ID in SEDE_imgs:
        vals.append(dists_unsafe_all[str(ID)][::len(dists_SEDE_all[str(ID)])])
        vals.append(dists_SEDE_all[str(ID)])
    vals2 = []
    for i in range(len(dists_SEDE_all[str(1)])):
        list1 = []
        for ID in SEDE_imgs:
            list1.append(dists_unsafe_all[str(ID)][i])
            list1.append(dists_SEDE_all[str(ID)][i])
        vals2.append(list1)
    print(len(cols), len(vals))
    bp = plt.boxplot(vals, labels=cols)
    for box in bp['boxes']:
        # change outline color
        box.set(color='#7570b3')
    plt.xticks(rotation=30)
    plt.grid(visible=True)
    plt.show()
    df = pd.DataFrame(vals2, columns=cols)
    ax = df.boxplot(grid='True', rot=30, color=colors)
    #ax = df.plot.box(grid='True', rot=30, color=colors)
    ax.set_ylabel('Heatmap Distances from RCC\'s medoid')
    ax.set_xlabel('RCC')

    ax.set_title(caseFile["caseStudy"])
    figure = (ax).get_figure()
    figure.savefig(join(os.getcwd(), "RQ2-3", "RQ2-"+caseFile["caseStudy"]+".pdf"), bbox_inches = "tight")
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    return
