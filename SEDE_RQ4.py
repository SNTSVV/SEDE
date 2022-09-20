import HeatmapModule
from imports import torch, join, os, stat, isfile, exists, makedirs, np, imageio, time, pd, random
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
import searchModule, assignModule
from searchModule import doImage, generateAnImage, setX_2, setX, nVar, generateHuman, getParamVals, getParamVals_2

def evaluateResults(caseFile):
    cID = input("Enter RCC ID:")
    csvPath = join(caseFile["filesPath"], "GeneratedImages", cID, "results.csv")
    cFile = join(caseFile["filesPath"], "GeneratedImages", cID, "config.pt")

    if int(caseFile["iee_version"]) == 2:
        paramNameList, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
            getParamVals_2()

    else:
        paramNameList, _, _, _, _, _, _, _, _, _, _ = getParamVals()
    paramDict = collectFailingMinMax(csvPath, paramNameList)
    cPART = collectRules(cFile)
    cPART = refineRulesMinMax(paramDict, cPART, paramNameList)
    generateImages(cID, caseFile, cPART, paramNameList, paramDict)


def collectRules(cFile):
    if isfile(cFile):
        cPART = torch.load(cFile)
    else:
        cPART = {}
        x = int(input("Enter # of rules:.. "))
        portions = []
        rules = []
        val1x = []
        val2x = []
        if x == 0:
            portions.append([1.0])
            rules.append([1e9])
        for j in range(0, x):

            portion = float(input("Enter portion (x/y):.. "))
            ruleType = int(input("Enter type of rule (1: plain - 2: sub-rule"))
            if ruleType == 1:
                n = input("Enter # of params:.. ")
                val1 = []
                val2 = []
                param = []
                for i in range(0, int(n)):
                    #param.append(input(
                    #    "Choose param: 1-3: cam_dir -- 4-6: cam_loc -- 7-9: lamp_loc -- 10-12: head_pose -- 13: face_model:.. "))
                    param.append(input(
                        "Choose param: 1-3: head -- 4-6: lamp_col -- 7-9: lamp_loc -- 10-12: lamp_dir -- 13: cam "
                        "-- 14: age -- 15: hue -- 16: iris -- 17: sat -- 18: val -- 19: freckle -- 20: oil -- 21: veins"
                        " -- 22: eye_col -- 23: gender :.. "))
                    val1.append(input("Enter Unsafe Value1:.. "))
                    val2.append(input("Enter Unsafe Value2:.. "))
                val1x.append(val1)
                val2x.append(val2)
                rules.append(param)
                portions.append(portion)
            elif ruleType == 2:
                n = int(input("Enter # of sub-rules"))
                for i in range(0, n):
                    p = input("Enter # of params:.. ")
                    val1 = []
                    val2 = []
                    param = []
                    for i in range(0, int(p)):
                        #param.append(input(
                        #    "Choose param: 1-3: cam_dir -- 4-6: cam_loc -- 7-9: lamp_loc -- 10-12: head_pose -- 13: face_model:.. "))

                        param.append(input(
                            "Choose param: 1-3: head -- 4-6: lamp_col -- 7-9: lamp_loc -- 10-12: lamp_dir -- 13: cam "
                            "-- 14: age -- 15: hue -- 16: iris -- 17: sat -- 18: val -- 19: freckle -- 20: oil -- 21: veins"
                            " -- 22: eye_col -- 23: gender :.. "))
                        val1.append(input("Enter Unsafe Value1:.. "))
                        val2.append(input("Enter Unsafe Value2:.. "))
                    val1x.append(val1)
                    val2x.append(val2)
                    rules.append(param)
                    portions.append(portion/n)

        cPART['rules'] = rules
        cPART['portions'] = portions
        cPART['val1x'] = val1x
        cPART['val2x'] = val2x
        torch.save(cPART, cFile)
    return cPART

def collectFailingMinMax(csvPath, paramNameList):
    imageList = pd.read_csv(csvPath)
    paramDict = {}
    print(paramNameList)
    print(nVar)
    for j in range(0, nVar):
        paramDict[paramNameList[j]] = []
    for index, row in imageList.iterrows():
        if not row["DNNResult"]:
            for i in range(0, nVar):
                paramDict[paramNameList[i]].append(float(row[paramNameList[i]]))
    return paramDict

def refineRulesMinMax(paramDict, cPART, paramNameList):
    paramNameListX = ["HeadPose_X", "HeadPose_Y", "HeadPose_Z", "LampCol_R", "LampCol_G",
     "LampCol_B", "LampLoc_X", "LampLoc_Y", "LampLoc_Z", "LampDir_X", "LampDir_Y", "Lamp_DirZ", "CamHeight",
     "Age", "Pupil_Size", "Iris_Size", "Eye_Sat", "Eye_Val", "Freckles", "Oil", "Veins", "EyeCol", "Gender"]
    paramFlag = False
    n = 0
    for param in paramDict:
        for j in range(0, len(cPART['rules'])):
            for i in range(0, len(cPART['rules'][j])):
                if (int(cPART['rules'][j][i]) - 1) < len(paramNameList):
                    if param == paramNameList[int(cPART['rules'][j][i]) - 1]:
                        print("R" + str(j) + ":", str(cPART['val1x'][j][i])[0:6], "<", paramNameListX[n], "<",
                              str(cPART['val2x'][j][i])[0:6])
                        if float(cPART['val1x'][j][i]) <= -1e8:
                            cPART['val1x'][j][i] = min(paramDict[param])
                        if float(cPART['val2x'][j][i]) >= 1e8:
                            cPART['val2x'][j][i] = max(paramDict[param])

                    else:
                        paramFlag = True

        if paramFlag or len(cPART['rules'][0]) == 0:
            print(str(min(paramDict[param]))[0:6], "<", paramNameListX[n], "<", str(max(paramDict[param]))[0:6])
        n += 1

    return cPART
def generateImages(cID, caseFile, cPART, paramNameList, paramDict):

    eval_imgs = input("Enter # of evaluation images:.. ")
    caseFile["SimDataPath"] = join(caseFile["filesPath"], "Pool")
    outDir = join(caseFile["filesPath"], "Evaluation", cID)
    if not exists(outDir):
        makedirs(outDir)
    f_ = 0
    clusterImages = []
    n = 0
    totalimgs = 0
    DNNResult = None
    t = time.time()
    print(cPART['rules'][0][0])
    for j in range(0, len(cPART['rules'])):
        toEval = int(cPART['portions'][j] * int(eval_imgs))
        total = 0
        while total < toEval:
            if int(caseFile["iee_version"]) == 2:
                x = setX_2(1, "R")
            else:
                x = setX(1, "R")
            print(total, DNNResult, str(time.time() - t)[0:5], end="\r")
            t = time.time()
            x = setNewX(x, paramDict, paramNameList, cPART['rules'][j], cPART['val1x'][j], cPART['val2x'][j])
            # if int(poseMode) == 2:
            #    x[19] = random.randint(0, 8)
            # elif int(poseMode) == 1:
            #    x[19] = random.randint(0, 2)
            if int(caseFile["iee_version"]) == 2:
                imgPath, F = generateHuman(x, caseFile)
            else:
                imgPath, F = generateAnImage(x, caseFile)
            if not F:
                f_ += 1
                # toEval += 1
            else:
                imgPath += ".png"
                N, DNNResult, P, L, D, _ = doImage(imgPath, caseFile, None)
                # DNNResult2, pred = testModelForImg(caseFile["DNN2"], L, imgPath, caseFile)
                if DNNResult:
                    n += 1
                # if DNNResult2:
                #    n2 += 1
                total += 1
                totalimgs += 1
                clusterImages.append(imageio.imread(imgPath))
    imageio.mimsave(join(outDir, "HUDD_" + str(len(clusterImages)) + "_" + str(100 * (n / totalimgs))[0:5] + '.gif'),
                    clusterImages)
    print("Accuracy:", 100 * (n / (totalimgs)))
    # print("Accuracy:", 100* (n2/(total)))
    print("not found", f_)
    if int(caseFile["iee_version"]) == 2:
        conversions = np.array([2343, int(n)])
        clicks = np.array([2750, int(totalimgs)])
        zscore, pvalue = proportions_ztest(conversions, clicks, alternative='two-sided')
        print('zscore = {:.4f}, pvalue = {:.4f}'.format(zscore, pvalue))
        ob_table = np.array([[2750, 2343], [int(totalimgs), int(n)]])
        result = stats.chi2_contingency(ob_table, correction=False)  # correction = False due to df=1
        chisq, pvalue = result[:2]
        print('chisq = {}, pvalue = {}'.format(chisq, pvalue))
    else:
        conversions = np.array([1914, int(n)])
        clicks = np.array([2200, int(totalimgs)])
        zscore, pvalue = proportions_ztest(conversions, clicks, alternative='two-sided')
        print('zscore = {:.4f}, pvalue = {:.4f}'.format(zscore, pvalue))
        ob_table = np.array([[2200, 1914], [int(totalimgs), int(n)]])
        result = stats.chi2_contingency(ob_table, correction=False)  # correction = False due to df=1
        chisq, pvalue = result[:2]
        print('chisq = {}, pvalue = {}'.format(chisq, pvalue))


def setNewX(x, paramDict, paramNameList, param, val1, val2):
    for j in range(0, nVar):
        minVal = min(paramDict[paramNameList[j]])
        maxVal = max(paramDict[paramNameList[j]])
        #if j == (nVar - 1):
        #    maxVal += 0.99  # we round down the facemodel value
        x[j] = random.uniform(minVal, maxVal)
        for z in range(0, len(param)):
            if j == (int(param[z]) - 1):
                #if j == nVar - 1:
                #    val2[z] = val2[z] + 0.99
                if float(val1[z]) < minVal:
                    val1[z] = minVal
                if float(val2[z]) > maxVal:
                    val2[z] = maxVal
                # print(val1[z], val2[z])
                if float(val1[z]) > float(val2[z]):
                    "error in parameters settings"
                x[j] = random.uniform(float(val1[z]), float(val2[z]))

        # print(paramNameList[j], x[j])
    return x


def Precision_Recall():
    # PR = input("Precision/Recall?: Y/N")
    # poseMode = input("Enter Evaluation mode (1: TrainingSet - 2: TestSet):")
    # csv2 = join(caseFile["filesPath"], "DT_MC_CC.csv")
    # csv3 = join(caseFile["filesPath"], "DT_RCC_CC.csv")
    # iL = pd.read_csv(csv2)
    # iL2 = pd.read_csv(csv3)
    pL = ["cam_look_direction_0", "cam_look_direction_1", "cam_look_direction_2", "cam_loc_0", "cam_loc_1", "cam_loc_2",
          "", "", "", "", "", "", "", "", "", "", "head_pose_0", "head_pose_1", "head_pose_2"]
    RM = 0
    RR2 = 0
    A = 0
    M = 0
    R2 = 0
    R = 0
    # if PR == "Y":
    #    for index, row in iL.iterrows():
    #        A += 1
    #        if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #            R += 1
    #        if int(row["clusterID"]) != 0:
    #            M += 1

    #            if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                    (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                    (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #                RM += 1
    #    for index, row in iL2.iterrows():
    #        if int(row["clusterID"]) == int(cID):
    #            R2 += 1
    #            if (float(val1[0]) <= float(row[str(pL[int(param[0]) - 1])]) <= float(val2[0])) and \
    #                    (float(val1[1]) <= float(row[str(pL[int(param[1]) - 1])]) <= float(val2[1])) and \
    #                    (float(val1[2]) <= float(row[str(pL[int(param[2]) - 1])]) <= float(val2[2])):
    #                RR2 += 1
    #    print(A, M,  R, RM, R2, RR2)
    #    print("MC_Precision:", 100* RM/R)
    #    print("MC_Recall:", 100 * RM/M)
    #    print("RCC_Precision:", 100* RR2/R)
    #    print("RCC_Recall:", 100 * RR2/R2)
    return