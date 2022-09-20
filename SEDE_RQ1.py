
from imports import join, np, plt, pd, argparse, os
import scipy.stats as stats
import statistics
from scipy.integrate import simpson
from numpy import trapz

import itertools as it
from bisect import bisect_left
from typing import List

from pandas import Categorical
ext = ".png"

def doRQ(caseStudy, folderPath):
    algos = ['PaiR', 'DeepNSGA-II', 'NSGA-II']
    div = []
    indv = []
    cls = []
    auc = []
    if caseStudy == "FLD":
        maxHours = 40
        clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yLim = [-0.01, 0.35]
    elif caseStudy == "HPD-F":
        maxHours = 40
        clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        yLim = [-0.0025, 0.12]
    else: #HPD-H
        maxHours = 200
        clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        yLim = [-0.01, 0.4]

    inputPath = join(folderPath, caseStudy)

    for algo in algos:
        div_dict, indv_dict, cls_dict, auc_list = doAlgorithm(algo, inputPath, clusters, maxHours)
        div.append(div_dict)
        indv.append(indv_dict)
        cls.append(cls_dict)
        auc.append(auc_list)

    #plotResultsAlgos(div[0], div[2], div[1], "Weighted Pairwise Chromosome Distances", yLim,
    plotResultsAlgos(div[0], div[2], div[1], "Pairwise Chromosome Distances", yLim,
                     join(inputPath, "RQ1-1-" + caseStudy + ext), maxHours, caseStudy)
    reportStats(div[0], div[1], div[2], auc, algos, join(inputPath, "diversity.txt"))

    plotResultsAlgos(indv[0], indv[2], indv[1], "% of individuals belonging to each RCC", [-5, 130],
                     join(inputPath, "RQ1-2-" + caseStudy + ext), maxHours, caseStudy)
    reportStats(indv[0], indv[1], indv[2], None, algos, join(inputPath, "individuals.txt"))

    plotResultsAlgos(cls[0], cls[2], cls[1], "% of covered clusters", [-5, 130],
                     join(inputPath, "RQ1-3-" + caseStudy + ext), maxHours, caseStudy)
    reportStats(cls[0], cls[1], cls[2], None, algos, join(inputPath, "clusters.txt"))

def doAlgorithm(algo, dirPath, clusters, maxHours):
    print("Computing", algo)
    div_dict, ind_dict, cls_dict, auc = collectData(clusters, dirPath, maxHours, algo)
    plotResults(div_dict, "Weighted Pairwise Chromosome Distances", None, join(dirPath, algo + "_diversity" + ext),
                maxHours, algo)
    plotResults(ind_dict, "% of individuals belonging to each RCC", [-5, 105],
                join(dirPath, algo + "_individuals" + ext), maxHours, algo)
    plotResults(cls_dict, "% of covered clusters", [-5, 105], join(dirPath, algo + "_clusters" + ext), maxHours, algo)
    return div_dict, ind_dict, cls_dict, auc

def collectData(clusters, inputPath, maxHours, algo):
    final_diversity_dict = {}
    final_individuals_dict = {}
    final_clusters_dict = {}
    auc = []
    for j in range(0, maxHours):
        final_diversity_dict[str(j)] = []
        final_individuals_dict[str(j)] = []
        final_clusters_dict[str(j)] = []
    for ID in clusters:
        for idx in range(1, 5):
            auc_list = []
            dirPath = join(inputPath, "RCC-" + str(ID), algo)
            clusters_list = np.load(join(dirPath, "Run-" + str(idx), "clusters.npy"), allow_pickle=True)
            diversity_list = np.load(join(dirPath, "Run-" + str(idx), "diversity.npy"), allow_pickle=True)
            individuals_list = np.load(join(dirPath, "Run-" + str(idx), "individuals.npy"), allow_pickle=True)
            diversity_dict = dict(enumerate(diversity_list.flatten(), 1))[1]
            individuals_dict = dict(enumerate(individuals_list.flatten(), 1))[1]
            clusters_dict = dict(enumerate(clusters_list.flatten(), 1))[1]
            for entry in diversity_dict:
                diversity_dict[entry] = diversity_dict[entry] * (float(individuals_dict[entry])/100)
                auc_list.append(diversity_dict[entry])
            auc.append(trapz(np.array(auc_list), dx=1))
            def append_to_final_results(result, final_dict):
                for entry in final_dict:
                    final_dict[entry].append(result[entry])
                return final_dict
            final_diversity_dict = append_to_final_results(diversity_dict, final_diversity_dict)
            final_individuals_dict = append_to_final_results(individuals_dict, final_individuals_dict)
            final_clusters_dict = append_to_final_results(clusters_dict, final_clusters_dict)

    return final_diversity_dict, final_individuals_dict, final_clusters_dict, auc

def plotResults(resultDict, yText, yLim, outputPath, maxHours, title):

    xText = "execution time (hrs)"
    c = []
    davg = []
    dmax = []
    dmin = []
    for i in range(0, maxHours):
        c.append(i+1)
        if len(resultDict[str(i)]) > 0:
            davg.append(sum(resultDict[str(i)]) / len(resultDict[str(i)]))
            dmax.append(max(resultDict[str(i)]))
            dmin.append(min(resultDict[str(i)]))
        else:
            davg.append(0.0)
            dmax.append(0.0)
            dmin.append(0.0)
    plt.plot(c, dmax, "g:")
    plt.plot(c, davg)
    plt.plot(c, dmin, "r:")
    plt.legend(['Max.', 'Avg.', 'Min.'])
    plt.ylabel(yText)
    plt.xlabel(xText)
    plt.ylim(yLim)
    plt.title(title)
    plt.savefig(outputPath)
    plt.cla()
    plt.clf()

def plotResultsAlgos(resultDict1, resultDict2, resultDict3, yText, yLim, outputPath, maxHours, title):
    c = list(range(1, maxHours+1))
    xText = "execution time (hrs)"
    def appendResults(result, maxHours):
        davg, dmin, dmax = [], [], []
        for i in range(0, maxHours):
            if len(result[str(i)]) > 0:
                davg.append(sum(result[str(i)]) / len(result[str(i)]))
                dmin.append(min(result[str(i)]))
                dmax.append(max(result[str(i)]))
            else:
                davg.append(0.0)
                dmin.append(0.0)
                dmax.append(0.0)
        return davg, dmin, dmax

    davg1, dmin1, dmax1 = appendResults(resultDict1, maxHours)
    davg2, dmin2, dmax2 = appendResults(resultDict2, maxHours)
    davg3, dmin3, dmax3 = appendResults(resultDict3, maxHours)

    if davg1[-1] == davg2[-1]:
        alpha1 = alpha1_1 = alpha1_3 = 0.7
        alpha2 = alpha2_1 = alpha2_3 = 0.5
        alpha3 = alpha3_1 = alpha3_3 = 1
    elif davg1[-1] == davg3[-1]:
        alpha1 = alpha1_1 = alpha1_3 = 0.7
        alpha2 = alpha2_1 = alpha2_3 = 1
        alpha3 = alpha3_1 = alpha3_3 = 0.5
    elif davg2[-1] == davg3[-1]:
        alpha1 = alpha1_1 = alpha1_3 = 1
        alpha3 = alpha3_1 = alpha3_3 = 0.7
        alpha2 = alpha2_1 = alpha2_3 = 0.5
    else:
        alpha1 = alpha2 = alpha3 = 1
        alpha1_1 = alpha2_1 = alpha3_1 = 0.7
        alpha1_3 = alpha2_3 = alpha3_3 = 0.5

    if float(davg1[-1]) == float(davg2[-1]) and float(davg1[-1]) == float(davg3[-1]):
        alpha1 = alpha1_1 = alpha1_3 = 0.7
        alpha3 = alpha3_1 = alpha3_3 = 0.6
        alpha2 = alpha2_1 = alpha2_3 = 0.5

    plt.plot(c, dmax1, "b:", label = "max(PaiR)", alpha=alpha1_1)
    plt.plot(c, davg1, "b", label = "PaiR", alpha = alpha1)
    plt.plot(c, dmin1, "b--", label= "min(PaiR)", alpha = alpha1_3)
    plt.plot(c, dmax3, "r:", label= 'max(DeepNSGA-II)', alpha = alpha3_1)
    plt.plot(c, davg3, "r", label="DeepNSGA-II", alpha = alpha3)
    plt.plot(c, dmin3, "r--", label="min(DeepNSGA-II)", alpha = alpha3_3)
    plt.plot(c, dmax2, "g:", label="max(NSGA-II)", alpha = alpha2_1)
    plt.plot(c, davg2, "g", label = "NSGA-II", alpha = alpha2)
    plt.plot(c, dmin2, "g--", label="min(NSGA-II)", alpha = alpha2_3)
    plt.legend(ncol = 3)
    plt.ylabel(yText)
    plt.xlabel(xText)
    plt.ylim(yLim)
    if yLim[-1] > 100:
        plt.yticks([0, 20, 40, 60, 80, 100])
    plt.title(title)
    plt.savefig(outputPath)
    plt.cla()
    plt.clf()

def reportStats(dict1, dict2, dict3, auc, algos, filePath):
    print(filePath)
    open(filePath, "w")
    file = open(filePath, "a")
    maxHourRange = min(len(dict1), len(dict2), len(dict3))

    for i in range(0, maxHourRange):
        if len(dict1[str(i)]) > 0 and len(dict2[str(i)]) > 0 and len(dict3[str(i)]) > 0:
            maxRange = min(len(dict1[str(i)]), len(dict2[str(i)]), len(dict3[str(i)]))
            VDA1, magnitude1 = VD_A((dict1[str(i)])[0:maxRange], (dict2[str(i)])[0:maxRange])
            VDA2, magnitude2 = VD_A((dict1[str(i)])[0:maxRange], (dict3[str(i)])[0:maxRange])
            stat1, UTest1 = stats.mannwhitneyu(dict1[str(i)][0:maxRange], dict2[str(i)][0:maxRange])
            stat2, UTest2 = stats.mannwhitneyu(dict1[str(i)][0:maxRange], dict3[str(i)][0:maxRange])
        else:
            VDA1, VDA2 = 0, 0
            UTest1, UTest2 = 0, 0
        APPR_avg = sum(dict1[str(i)])/len(dict1[str(i)]) if len(dict1[str(i)]) > 0 else 0.0
        APPR_std = statistics.stdev(dict1[str(i)]) if len(dict1[str(i)]) > 0 else 0.0
        BL1_avg = sum(dict2[str(i)])/len(dict2[str(i)]) if len(dict2[str(i)]) > 0 else 0.0
        BL1_std = statistics.stdev(dict2[str(i)]) if len(dict2[str(i)]) > 0 else 0.0
        BL2_avg = sum(dict3[str(i)])/len(dict3[str(i)]) if len(dict3[str(i)]) > 0 else 0.0
        BL2_std = statistics.stdev(dict3[str(i)]) if len(dict3[str(i)]) > 0 else 0.0
        file.write("hour: " + str(i+1) + "\n")
        file.write("significance (U-test):" + algos[0] + " / " + algos[1] + " " + str(UTest1) + "\n")
        file.write("significance (U-test):" + algos[0] + " / " + algos[2] + " " + str(UTest2) + "\n")
        file.write("VDA: " + algos[0] + " / " + algos[1] + " " + str(VDA1) + "\n")
        file.write("VDA: " + algos[0] + " / " + algos[2] + " " + str(VDA2) + "\n")
        file.write(algos[0] + " Datapoints" + " " + str(dict1[str(i)]) + "\n")
        file.write(algos[1] + " Datapoints" + " " + str(dict2[str(i)]) + "\n")
        file.write(algos[2] + " Datapoints" + " " + str(dict3[str(i)]) + "\n")
        file.write(algos[0] + " Avg." + " " + str(APPR_avg) + " ## " + algos[1] + " Avg." + " " + str(BL1_avg) +
                   " ## " + algos[2] + " Avg." + " " + str(BL2_avg) + "\n")
        file.write(algos[0] + " STD." + " " + str(APPR_std) + " ## " + algos[1] + " STD." + " " + str(BL1_std) +
                   " ## " + algos[2] + " STD." + " " + str(BL2_std) + "\n")
        file.write("********************************************************* \n")
        print_stats(i+1, UTest1, UTest2, VDA1, VDA2, APPR_avg, BL1_avg, BL2_avg, APPR_std, BL1_std, BL2_std)
    file.close()

    if auc is not None:
        VDA_auc1, magnitude_auc1 = VD_A(auc[0], auc[1])
        VDA_auc2, magnitude_auc2 = VD_A(auc[0], auc[2])
        stat_auc1, u_auc1 = stats.mannwhitneyu(auc[0], auc[1])
        stat_auc2, u_auc2 = stats.mannwhitneyu(auc[1], auc[2])
        print("AUC", algos[0], algos[1], u_auc1, VDA_auc1)
        print("AUC", algos[0], algos[2], u_auc2, VDA_auc2)


def print_stats(hour, u1, u2, vda1, vda2, appr_avg, bl_avg, bl2_avg, appr_std, bl_std, bl2_std):
    text = " & \multicolumn{1}{c|}{"
    end = "}"
    newline2 = "\ \hline"
    newline1 = "\ "
    newline1 = newline1[0]
    if appr_avg > 10:
        perc = "\%"
        dec = '.2f'
    else:
        perc = ""
        dec = '.3f'

    if ((hour%5 == 0) and hour < 41) or ((hour%25==0) and hour > 24):
        print(f'{hour}{text}{appr_avg:{dec}}{perc} ({appr_std:{dec}}){end}'
              f'{text}{bl2_avg:{dec}}{perc} ({bl2_std:{dec}}){end}'
              f'{text}{bl_avg:{dec}}{perc} ({bl_std:{dec}}){end}'
              f'{text}{u2}{end}{text}{vda2:{dec}}{end}'
              f'{text}{u1}{end}{text}{vda1:{dec}}{end} '
              f'{newline1+newline2}\n')
        #print("hour:", hour)
        #print("significance (U-test):", u)
        #print("VDA: " + " " + str(vda))
        #print("APPR Avg.", appr_avg)
        #print("BL Avg.", bl_avg)
        #print("APPR STD.", appr_std)
        #print("BL STD.", bl_std)
        #print("*********************************************************")


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = stats.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    #A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEDE_RQ1')
    parser.add_argument('-case', '--caseStudy', default="FLD", help='HPD-H, HPD-F, FLD', required=False)
    args = parser.parse_args()
    doRQ(args.caseStudy, join(os.getcwd(), "RQ1"))