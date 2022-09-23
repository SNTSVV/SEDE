# SEDE (Simulator-based Explanations of DNN-Errors)

## Introduction

SEDE is a tool that is used to generate explanations for DNN errors with any simulator.
SEDE is a cluster-based approach that relies on the clusters generated by [HUDD](https://github.com/SNTSVV/HUDD-Toolset), but might be used with any generated clusters.

This package provides the images generated by our approach and competing approaches appearing in ["Simulator-based explanation and debugging of hazard-triggering events in DNN-based safety-critical systems"](https://arxiv.org/abs/2204.00480) by Hazem Fahmy, Fabrizio Pastore, Member, IEEE, and Lionel Briand, Fellow, IEEE, Thomas Stifter.

## Project Description

To support safety analysis practices, we propose SEDE, a technique that generates readable descriptions for commonalities in failure-inducing, real-world images and improves the DNN through effective retraining. SEDE leverages the availability of simulators, which are commonly used for cyber-physical systems. It relies on genetic algorithms to drive simulators towards the generation of images that are similar to failure-inducing, real-world images in the test set; it then employs rule learning algorithms to derive expressions that capture  commonalities in terms of  simulator parameter values. The derived expressions are then used to generate additional images to retrain and improve the DNN. With DNNs performing in-car sensing tasks, SEDE successfully characterized hazard-triggering events leading to a DNN accuracy drop. Also, SEDE enabled retraining leading to significant improvements in DNN accuracy, up to 18 percentage points.

## NOTICE on dependencies and libraries

SEDE is a toolset that might be used to generate explanations for DNN errors with any simulator.
However, for our experiments we rely on the faces simulator developed by IEE S.A. (https://iee-sensing.com)
We refer to such simulator as IEE-Face-Simulator.
The released implementation of SEDE invokes the IEE-Face-Simulator through a program call (i.e., not through APIs).
SEDE makes use of two versions of IEE-Face-Simulator, v0.5, which is in directory IEE_V1, and v1.0, which is in directory IEE_V2.

Both the two versions of the IEE-Face-Simulator are released with GPL v3 licence (see files 'gpl-3.0.txt' in both directories).
The directory IEEPackage is Copyright (C) 2018-2022 IEE S.A. (https://iee-sensing.com) released with GPL v3 licence.

## Usage

The package contains the python code implementation of the SEDE approach and the files used to evaluate RQ1-5 (SEDE_RQx.py) along with IEE simulators and DeepJanus adoption of our case studies. We modified the implementation of DeepJanus for their BeamNG case study with the following changes:
* individual.py: we do not deal with a pair of images but only one image; distance computation is performed directly in the archive class, where we import “doParamDist()” from “searchModule.py” 
* core.nsga2.py: 
  * the budget is provided as execution time, not number of generations 
  * ProblemClass.final_population is assigned with the population stored in the archive. The assignment occurs in “Archive.archive_metrics()”. This is slightly different from DeepJanus where the archive is stored inside BeamNGProblem. 
* core.archive_impl.py: the SmartArchive() function has been modified to add only individuals that are inside the cluster (F2 <=1); the distance function is replaced by ours (cosine parameters similarity) 

To run the tool you need to have the case studies setup in a separated directory and then generate HUDD clusters using the command

> python SEDE.py -o ./HPD -SEDE HUDD

To run SEDE, you need to specify the IEE-Face-Simulator version (options are 1 or 2) and the algorithm to use (options are SEDE or DeepJanus)

> python SEDE.py -o ./HPD -iee 1 -SEDE SEDE
> python SEDE.py -o ./HPD -iee 1 -SEDE DeepJanus

To run the evaluation you need to use the following command:

> python SEDE.py -SEDE RQ1

where RQx is the RQ to be evaluated (options are RQ1, RQ2, RQ3, RQ4, RQ5)
N.B.: for RQ1 you need to have the RQ1 directory shared with the replicability package inside the parent directory of this file


## Contents

Download the empirical data for RQ1-5 [here](https://figshare.com/s/a7fd8e713e038ee0d86c)

It contains the following folders for evaluation:
* 'SEDE Expressions' contains the PDF files with the expressions generated by SEDE for each case study DNN.
For each RCC, the expression contains the constraints for the rules generated by PART (in bold) and the constraints generated by SEDE for the simulator parameter values in common for both safe and unsafe images.

* 'RQ1' contains a folder for each case study DNN ('HPD-F', 'HPD-H', 'FLD'). Inside, you can find a folder for each RCC; 
such folder contains the real-world images belonging to the cluster along with the images generated by SEDE, NSGA-II 
and DeepNSGA-II in four different experiments.
E.g.: 'RQ1/HPD-F/RCC-1' contains 'SEDE' and 'NSGA-II' and ‘HUDD', where ‘HUDD’ contains the real-world images in the cluster generated by HUDD. For ‘FLD’, the clusters are not included in the package due to copyrights purposes. ‘SEDE’, ‘DeepNSGA-II’ and ‘NSGA-II’ contains folders named ‘Run-1’, ‘Run-2’, ‘Run-3’, and ‘Run-4’, where each contains the images generated in each run reported for RQ1. Inside each folder, you can find ‘diversity.npy’ which reports the diversity observed over time (RQ1.1), ‘indvidiuals.npy’ which reports the percentage of individuals observed over time (RQ1.2), and ‘clusters.npy’ which reports the percentage of covered clusters observed over time (RQ1.3).

* ‘RQ2-3’ contains a folder for each case study DNN (‘HPD-F’, ‘HPD-H’, ‘FLD’) with a folder for each RCC containing the images generated and evaluated by SEDE for these RQs. Since both RQ2 and RQ3 evaluate the same images generated by SEDE in Step 2.1, they are combined in a single folder.

* 'RQ4' contains a folder for each case study DNN ('HPD-F', 'HPD-H', 'FLD') with images generated according to SEDE expressions for each cluster and the obtained accuracy.
E.g.: 'RQ4/HPD-F/RCC-1_500_10.2%.gif' where 500 is the number of images and 10.2% is the obtained accuracy.

* 'RQ5' contains a folder for each case study DNN ('HPD-F', 'HPD-H', 'FLD'). Each case study DNN contains a folder with the images selected/generated by competing approaches ('HUDD', 'RBL') for the 10 runs.
E.g.: 'RQ5/HPD-F/HUDD/RCC-1.gif' contains the images selected by HUDD for Cluster 1.


## Reference:

If you use our work, please cite SEDE in your publications. Here is an example BibTeX entry:
```
@misc{https://doi.org/10.48550/arxiv.2204.00480,
  doi = {10.48550/ARXIV.2204.00480},
  url = {https://arxiv.org/abs/2204.00480},
  author = {Fahmy, Hazem and Pastore, Fabrizio and Briand, Lionel and Stifter, Thomas},
  keywords = {Software Engineering (cs.SE), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Simulator-based explanation and debugging of hazard-triggering events in DNN-based safety-critical systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
