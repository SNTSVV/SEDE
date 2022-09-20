# IEE-Face-Simulator

IEE-Face-Simulator - copyright IEE

## datatools: 

1. Generating synthetic data using blender and mhx2 models

    a. dependence: blender 2.79 + mhx2 plugin 

    b. examples:

        i.    python ieekeypoints.py # generating 2d images from Aaa02_o.mhx2, mimic the pose from IEEPackage/pose
            uses ieesimulator.py and ieeclass.py

        ii.  ieelabelling.py # creating keypoints-labeled 3d model: (1) open blendder; (2) load a 3d mhx2 model and switch to script mode, loading labelling.py; (3) mannully select keypoints, ieelabelling.py saves labeled data

2. Preparing training dataset using generated synthetic data

    a. dependence: dlib 

    b. examples:

       ii.   python ieedatavendor.py # generating training dataset. (1)./evidence demonstrates the correctness of: training data -- data labels; (2) trainig data saved to ieedata/training.npy

3. Training a model and predicting labels:

    i. using ieepredict.py and model.py

