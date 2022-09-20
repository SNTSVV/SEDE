"""
Copyright (C) 2018-2022 IEE S.A. (https://iee-sensing.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
created by: jun wang @ iee
"""
import sys
import pathlib as pl
import os
import random
import argparse
from os.path import basename
sys.path.append(pl.Path(".").resolve())
sys.path.insert(1,'./')
print(sys.path)
import IEE_V2.utils2 as utils2
import IEE_V2.ieesimulator2 as ieesim
from os.path import isfile
import IEE_V2.config_iee as cfg
from pathlib import Path
#import numpy as np


def generate(data):
    """
    Generate samples given the configuration in `data`.

    Parameters
    ----------
    data: list
        Contains the configuration parameters for the scene to generate.
    """
    pose_file, model_file, label_file,sample_type = data[0], data[1], data[2], data[3]
    width, height = data[4], data[5] #376 #240
    dst_datafolder = data[6]
    background_dir = data[7]
    subsampling = data[8]
    print("pose file", pose_file)
    print("model file", model_file)
    print("label file", label_file)
    print(f"###### ---- USING SUBSAMPLING 0: {subsampling}")
    #print("CFG: dst", cfg.dst_datafolder)
    print("imgPath", imgPath)
    if sample_type == 1: # basic
        imitator = ieesim.IEEImitator(model_file,label_file,pose_file,\
               #width,height, dst_datafolder=cfg.dst_datafolder, constain_face=cfg.constain_face, GPUs=cfg.GPUs, background_dir=background_dir, subsampling=subsampling)
               width,height, dst_datafolder=imgPath, constain_face=cfg.constain_face, GPUs=cfg.GPUs, background_dir=background_dir, subsampling=subsampling)
    elif sample_type == 2:# hand on face
        imitator = ieesim.IEEImitatorHandonFace(model_file,label_file,pose_file,\
               width,height, dst_datafolder=cfg.dst_datafolder,constain_face=cfg.constain_face, GPUs=cfg.GPUs, background_dir=background_dir, subsampling=subsampling)
    elif sample_type == 3:# rnd_env
        imitator = ieesim.IEEImitatorRndEnv(model_file,label_file,pose_file,\
               width,height, dst_datafolder=cfg.dst_datafolder,constain_face=cfg.constain_face, GPUs=cfg.GPUs)
    elif sample_type == 4:# for future use
        imitator = ieesim.IEEImitator(model_file,label_file,pose_file,\
               width,height, dst_datafolder=cfg.dst_datafolder,constain_face=cfg.constain_face, GPUs=cfg.GPUs)
    else:
        print("WARN: IEESimulator does not support this sampl type: ", sample_type)
        return

    #imitator.create_samples()
    imitator.create_samples_allParams(dst_datafolder)
    return

class IEEKPgenerator(object):
    def __init__(self, model_folder, pose_folder, label_folder=None, background_dir=None, subsampling=1):
        """
        Generate images of models and store the key point information for each image.

        Parameters
        ----------
        model_folder: pl.Path
            Path the folder containing the human models.
        pose_folder: pl.Path
            Path containing the pose data
        label_folder: pl.Path
            Path containing the labels (if the location is different to `model_folder`)
        background_dir: pl.Path
            If backgrounds are requested, i.e. not `None`, the location where the HDR backgrounds are stored.
            A random background from this directory will be chosen from this directory.
        subsampling: int
            Only use poses whose identifier satisfies `identifier % subsampling = 0`. This is used to reduce the number of generated images.
        """
        self.model_folder = model_folder
        self.pose_folder = pose_folder
        self.label_folder = label_folder
        self.model_pose_labels = self.get_model_pose_labels()
        self.background_dir = background_dir
        self.subsampling = subsampling
        return
        
    def get_model_pose_labels(self):
        """
        TODO: This depends on the global variable cfg. Remove this!
        """
        if cfg.org_type==1:
            return self.get_model_pose_labels_1()
        elif cfg.org_type==2:
            return self.get_model_pose_labels_2()
        else:
            print("WARN: no legal models files.")
            return None
    
    #add a shuffle to the file lists?
    def get_model_pose_labels_2(self):
        """
        Model files and label files are located in the same directory.
        """
        #all_mod_files = utils.get_files(self.model_folder, cfg.model_type)
        all_mod_lab_files = utils2.get_model_label_files(self.model_folder, cfg.model_type, "landmarks.npy")
        all_pose_files = utils2.get_files(self.pose_folder, ".csv")
        print("Model Lab Files", all_mod_lab_files)
        print("Pose Files", all_pose_files)
        #when model# is more than pose#, no rnd shuffle here, in case of sth like reproduce
        model_num = len(all_mod_lab_files)
        pose_num = len(all_pose_files)
        print(f"-- number of poses: {pose_num}--")
        if model_num > pose_num:
            all_pose_files = all_pose_files+all_pose_files[:model_num-pose_num]
        num_to_simu = model_num #min(len(all_pose_files), len(model_label_pairs))
        #num_to_simu = 10
        print(f"----- Simulating {num_to_simu} models.")

        if num_to_simu < 1:
            return None

        model_label_pairs = all_mod_lab_files[:num_to_simu]
        model_files, label_files = zip(*all_mod_lab_files)
        all_pose_files = all_pose_files[:num_to_simu]
        z = zip(all_pose_files, model_files, label_files)
        c = list(z)
        random.shuffle(c)
        return c

    #add a shuffle to the file lists?
    def get_model_pose_labels_1(self):
        """
        Model files and label files are located in different directories.
        """
        all_mod_files = utils.get_files(self.model_folder, cfg.model_type)
        all_pose_files = utils.get_files(self.pose_folder, ".csv")

        def match_pose_label(all_mod_files, label_folder):
            model_label_pairs = []
            for model_file in all_mod_files:
                #print(model_file, model_file.split(".obj")[0]+".isomap.png")
                model_who = model_file.split("/")[-1]
                model_who = model_who.split(".")[0]
                label_file = os.path.join(label_folder,model_who.lower()+"_label.npy")
                if isfile(label_file):
                    utils.print("matched label file: ", label_file)
                    model_label_pairs.append([model_file,label_file])
            return  model_label_pairs

        model_label_pairs = match_pose_label(all_mod_files,self.label_folder)
        
        #when model# is more than pose#, no rnd shuffle here, in case of sth like reproduce
        model_num = len(model_label_pairs)
        pose_num = len(all_pose_files)
        if model_num > pose_num:
            all_pose_files = all_pose_files+all_pose_files[:model_num-pose_num]
        num_to_simu = model_num #min(len(all_pose_files), len(model_label_pairs))

        if num_to_simu < 1:
            return None

        model_label_pairs = model_label_pairs[:num_to_simu]
        model_files, label_files = zip(*model_label_pairs)
        all_pose_files = all_pose_files[:num_to_simu]
        
        return random.shuffle(list(zip(all_pose_files, model_files, label_files)))

    def form_parameters(self, model_pose_labels, img_width, img_height, dst_datafolder, background_dir=None,
                        subsampling=0):
        """
        Create the configuration list to be passed to the `generate` function.
        """
        pose_file, model_file, label_file = zip(*model_pose_labels)
        width_arr = [img_width]*len(pose_file)
        height_arr = [img_height]*len(pose_file)
        dsts = [dst_datafolder]*len(pose_file)
        sample_types = [cfg.sample_type]*len(pose_file)
        background_arr = [background_dir] * len(pose_file)
        subsampling_arr = [subsampling] * len(pose_file)
        all_paras = zip(pose_file,model_file,label_file, sample_types, width_arr,height_arr,dsts, background_arr,
                        subsampling_arr)
        return all_paras

    def generate_with_single_processor(self, img_width, img_height, dst_datafolder=None):
        if not self.model_pose_labels:
            print("no file exists")
            return
        all_paras = self.form_parameters(self.model_pose_labels, img_width, img_height, dst_datafolder, self.background_dir, self.subsampling)

        for idx, item in enumerate(all_paras):
            print(item)
            generate(item)


    def generate_with_multi_processor(self, img_width, img_height, dst_datafolder=None):
        if not self.model_pose_labels:
            print("no file exists")
            return

        import multiprocessing
        from multiprocessing import Pool

        cpu_count = multiprocessing.cpu_count()

        all_paras = self.form_parameters(self.model_pose_labels, img_width, img_height, dst_datafolder, self.background_dir, self.subsampling)
        all_paras = list(all_paras)

        for idx in range(len(all_paras)):
            print(all_paras[idx])

        # use_cpu = min(len(all_paras), cpu_count//4)
        use_cpu = min(len(all_paras), 4)
        use_cpu = max(1,use_cpu)

        if __name__ == '__main__':
            print("INFO: ", cpu_count, " CPUs exists.", use_cpu, "will be used.")
            pool = Pool(processes=use_cpu)
            try:
                pool.map(generate, all_paras)
            except Exception as e:
                pass
        return
if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='DNN debugger')
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    imgPath = argv[1]
    print("Generating an image", basename(imgPath))

    generator = IEEKPgenerator(Path(os.path.dirname(imgPath)), cfg.pose_folder, cfg.label_folder, cfg.background_dir, cfg.subsampling)
    if cfg.muti_processor:
        generator.generate_with_multi_processor(cfg.width, cfg.height)
    else:
        generator.generate_with_single_processor(cfg.width, cfg.height, imgPath)

    #generator.generate_with_multi_processor(width, height,"./test_img/")
    #generator.generate_with_single_processor(width, height, "./test_img/")














    
