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

def plotRQ(caseFile):
    layerClust = join(caseFile["filesPath"], "ClusterAnalysis_" + str(caseFile["clustMode"]),
                      caseFile["selectedLayer"] + ".pt")
    clsData = torch.load(layerClust, map_location=torch.device('cpu'))
    RQ3(clsData['clusters'], caseFile)
    return


def RQ3(clusters, caseFile):
    print("IEE Simulator V", caseFile["iee_version"])
    print("SEDE Evaluation based on simulator parameters reduction")

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
    dictVar_2 = {}
    dictLabel = {}
    for ID in clusters:
        print(ID)
        dictLabel[str(ID)] = []
        dictVar_2[str(ID)] = {}
        #path = join(GI, str(ID-7))
        path = join(GI, str(ID))
        dirPath = join(path, 'Pop', '1')
        #res = searchModule.loadCP(join(path, 'res1.pop'))
        imgs = []
        dir = os.listdir(join(path, 'Pop', '1'))
        res = searchModule.loadCP(join(path, "res1.pop"))
        if hasattr(res, "pop"):
            res = res.pop
        for ind in res:
            x = ind.X
            img = searchModule.processX(x)
            imgPath = join(outPath, img + ".png") #fixme
            imgPath = join(dirPath, img + ".png") #fixme
            #print(imgPath)
            x = []
            imgParams = (img.split(".png")[0]).split("_")
            for j in imgParams:
                x.append(float(j))
            if len(x) > searchModule.nVar:
                continue
            imgs.append(imgPath)
            #if not os.path.isfile(imgPath):
            #    if int(caseFile["iee_version"]) == 2:
            #        searchModule.generateHuman(x, caseFile)
            #    else:
            #        searchModule.generateAnImage(x, caseFile)
            #N, DNNResult, P, L, D3, layersHM = searchModule.doImage(imgPath, caseFile, None)
            #dictLabel[str(ID)].append(L)

        i = 0
        for param in paramList:
            dictVar_2[str(ID)][param] = []
            for img in imgs:
                x = []
                imgParams = (os.path.basename(img).split(".png")[0]).split("_")
                for j in imgParams:
                    x.append(float(j))
                if len(x) > searchModule.nVar:
                    continue
                dictVar_2[str(ID)][param].append(x[i]) #TODO: check if correct params
            i += 1
    resDict = {}
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    #vals = [[0.0] * 10] * 23
    vals = []
    for j in range(0, searchModule.nVar):
        vals.append([])
        for i in range(0, len(clusters)):
            vals[j].append(0.0)
    #print(vals)
    i = 0
    for ID in clusters:
        continue
        resDict[str(ID)] = []
        j = 0
        for param in dictVar_2[str(ID)]:
            varRed = stat.variance(dictVar_2[str(ID)][param])/stat.variance(dictVar[param])
            varRedPerc = 100*(1-varRed)
            varRedPerc = 0 if varRedPerc < 0 else varRedPerc
            #print(j, i)
            vals[j][i] = varRedPerc
            #if 0.0 <= varRed <= 0.1:
            #    count5 += 1
            #    break
            #if 0.1 < varRed <= 0.2:
            #    count4 += 1
            #    break
            #if 0.2 < varRed <= 0.3:
            #    count3 += 1
            #    break
            #if 0.3 < varRed <= 0.4:
            #    count2 += 1
            #    break
            #if 0.4 < varRed <= 0.5:
            #    count1 += 1
            #    break
                #print(ID, param, 100*(1-varRed), max(dictVar_2[str(ID)][param]), min(dictVar_2[str(ID)][param]), sum(dictVar_2[str(ID)][param])/len(dictVar_2[str(ID)][param]))
            #print(len(dictVar_2[str(ID)][param]), len(dictVar[param]))
            #print(dictVar_2[str(ID)][param])
            #print(dictVar[param])
            j += 1
        i += 1
    #print(vals[0])
    coloumnList = []
    #clusters = [1, 2, 4, 5, 6, 8, 10] #HPD-F
    #clusters = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #HPD-H
    for ID in clusters:
        coloumnList.append('RCC-'+str(int(ID)))
    df = pd.DataFrame(vals, columns=coloumnList)
    ax = df.plot.box(grid='True')
    ax.set_ylabel('Variance Reduction (%)')
    figure = (ax).get_figure()
    # figure.savefig(join(GI_path, "RQ4.png"))
    figure.savefig(join(GI, "RQ3-HPD2.png"))
    # figure.savefig(join(GI_path, "RQ4.pdf"))
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    #print(vals)

    print(count1, count2, count3, count4, count5)
    return dictLabel, SEDE_imgs