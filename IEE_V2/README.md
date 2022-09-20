# Generation of synthetic data

## Introduction

This directory provides the tools to generate new human face images based on MB-Lab.

## Setup

The generation of the models is performed using the MB-Lab Blender plugin.
For this reason Blender needs to be installed and the MB-Lab plugin enabled.

For reference, we use `Blender 2.81` and `MB-Lab 1.7.8.42`.

We steer MB-Lab through the `mblab-interface` Python package which is provided in a subdirectory of this distribution.
For instructi   ons on how to install it, take a look at the README.md in the corresponding directory.

## Generate 3D Models using MB-Lab

Start by moving from the root directory to the `mblab-interface` directory. In this directory, run the command

```sh
> cd mblab-interface
> python3 scripts/snt_data_generation_interface.py -n 10 --start-index 1 -g 0,1,2,3 --random-engine Noticeable
```

This generates `n=10` random human 3D models and stores them in the directory `../snt_simulator/data/mb_lab/run_X` where `X` is the number of the run. 

In addition we start the run labelling at `--start-index 1`.
If you have generated a first round of 10 models and want to generate another 10, you can just rerun the command by changing the start index to 11.

For each model a sample image is generated such that the models can be inspected visually.
To do so, the script is instructed to use the GPU's 0, 1, 2 and 3.

Finally, `--random-engine Noticeable` instructs the MB-Lab interface to generate random and noticeably different models.
Other options are:

- Light for small changes
- Realistic for more variation
- Noticeable for noticeable variations (this is the default)
- Caricature (looks often unrealistic)
- Extreme (looks often unrealistic)

In addition expressions and hair will be added randomly. Male characters can be bald.
To add hair, MB-Lab needs to load the hair cuts as assets and they are stored in the directory `../mblab_data_assets/assets/hair`.

## Run the simulator

To run the simulator go to the directory `snt_simulator` and run

```sh
> export PYTHONPATH=$PWD:$PYTHONPATH 
> python3 run_keypoints.py
```
