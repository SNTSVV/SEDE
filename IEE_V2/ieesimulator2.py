""""
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
# Modified by: Hazem Fahmy - hazem.fahmy@uni.lu - Added function create_samples_allParams()

import os
import sys
from os.path import isfile, join
import numpy as np
from IEE_V2.ieeclass2 import IEEPerson, IEE4DFacePerson, IEEScenario, IEEMbLabPerson
import IEE_V2.utils2 as utils2
from random import randint, uniform
import bpy

class IEESimulator(object):

    def __init__(self, model_file, label_file, img_width, img_height, GPUs=[0], background_dir=None):
        self.model_file = model_file
        self.model_type = model_file.split(".")[-1]
        self.label_file = label_file
        self.img_width = img_width
        self.img_height = img_height
        self.ensure_available()
        self.coods_3d = self.get_3d_coords_from_label()
        self.GPUs = GPUs
        self.background_dir = background_dir
        return

    # you can manipulate the 3D model here as you want
    def hook_ieeperson(self, person, cfg_to_person):
        return

    # you can manipulate the blender envirionment here as you want
    def hook_ieescenario(self, scenario, cfg_to_scenario):
        return

    def get_3d_coords_from_label(self):
        coods_3d = np.load(self.label_file,  allow_pickle=True)
        coods_3d = coods_3d.item()
        return coods_3d
   
    def customize_sample(self, cfg_to_person=None, cfg_to_scenario=None, dst=None, background=None):
        person = None
        if self.model_type == "mhx2":
            person = IEEPerson(self.model_file)
            #x=0
        elif self.model_type == "obj":
            person = IEE4DFacePerson(self.model_file)
        elif self.model_type == "blend":
            person = IEEMbLabPerson(self.model_file)
        
        #label_data = self.get_3d_coords_from_label()
        #if person.model_type=="blend":
        #   person.include_face_landmarks_blend(label_data)

        self.hook_ieeperson(person, cfg_to_person)
        scenario = IEEScenario(person=person, res_x=self.img_width, res_y=self.img_height, render_scale=1.0, label_data=self.coods_3d, GPUs=self.GPUs, background_dir=background)
        #print("gaze:", scenario._get_gaze())
        self.hook_ieescenario(scenario, cfg_to_scenario)
        if dst:
            scenario.save_3dmodel_to_2dsample(self.coods_3d, dst, True)
            person.clean()


    def ensure_available(self):
        vtrue = True
        if not isfile(self.model_file):
            print("cannot find pose file: ", self.model_file)
            vtrue = False

        if not isfile(self.label_file):
            print("cannot find pose file: ", self.label_file)
            vtrue = False

        assert vtrue
        assert self.img_width > 1 and self.img_height > 1


class IEEImitator(IEESimulator):
    def __init__(self, model_file, label_file, pose_file, img_width, img_height, dst_datafolder=None, sampling=False, constain_face=0, GPUs=[], background_dir=None, subsampling=1):
        super(IEEImitator,self).__init__(model_file, label_file,img_width, img_height,GPUs, background_dir)
        self.pose_file = pose_file
        assert isfile(self.pose_file)
        self.sampling = sampling
        self.constain_face = constain_face
        self.dst, self.pose_who, self.model_who = self.create_dst_datafolder(dst_datafolder)
        self.subsampling=subsampling
        print(f"###### ---- USING SUBSAMPLING: {subsampling}")

    def get_pose_data(self):
        pose_data = np.loadtxt(self.pose_file, delimiter=',', skiprows=1)
        pose_data = pose_data[:,[0, 10, 11, 12]].astype(np.float32)
        if self.sampling:
            #using only even rows pose data
            pose_data = pose_data[::2]
        print("pose_data: ", pose_data.shape)
        return pose_data

    def create_dst_datafolder(self, dst_datafolder):
        pose_who = self.pose_file.split("/")[-1]
        pose_who = pose_who.split(".")[0]
        model_who = self.model_file.split("/")[-1]
        model_who = model_who.split(".")[0]

        if not dst_datafolder:
            dst_datafolder = "./iee_imgs/"
        #print(model_who.split("\\")[0])
        #dst = dst_datafolder+model_who.split("\\")[0]+"/"+model_who.split("\\")[1]
        #print("dst_folder", dst_datafolder)
        #utils2.ensure_folder(dst)
        return os.path.dirname(dst_datafolder), pose_who, model_who

    def discovery_used_poses(self):
        all_png_files = utils2.get_files(self.dst, ".png")
        all_idxs = []
        for afile in all_png_files:
            tmp_str = afile.split(".png")[0]
            identifier = tmp_str.split("/")[-1]
            all_idxs.append(identifier)
        return set(all_idxs)


    # you can manipulate the 3D model here as you want
    def hook_ieeperson(self, person, cfg_to_person):
        x_ang, y_ang, z_ang = cfg_to_person
        person.adjust_face(x_ang, y_ang, z_ang)
        return
    # you can manipulate the blender envirionment here as you want
    def hook_ieescenario_params(self, scenario, cfg_to_scenario):
        scenario.setup()
        print(cfg_to_scenario)
        rnd_energy = cfg_to_scenario[0]
        color = (cfg_to_scenario[1], cfg_to_scenario[2], cfg_to_scenario[3])

        loc = (cfg_to_scenario[4], cfg_to_scenario[5], cfg_to_scenario[6])
        dir = (cfg_to_scenario[7], cfg_to_scenario[8], cfg_to_scenario[9])
        rnd_lamp = randint(0, 1)
        lamp_type = "SUN"
        if rnd_lamp == 1:
            lamp_type = "AREA"
        scenario.set_lamp(energy=rnd_energy, color=color, lamp_type=lamp_type, location=loc, direct=dir)
        scenario.set_camera(target=[0, 0, cfg_to_scenario[10]])
        return

    def hook_ieescenario(self, scenario, cfg_to_scenario):
        scenario.setup()
        rnd_energy = uniform(0.5,1.5)
        rnd_color = randint(0,2)
        color = (1.0, 1.0, 1.0)
        if rnd_color == 0:
            color = (1.0, 0, 0)
        elif rnd_color == 1:
            color = (0, 1.0, 0)
        else:
            color = (0, 0, 1.0)
        rnd_lamp = randint(0,1)
        lamp_type = "SUN"
        if rnd_lamp == 1:
            lamp_type = "AREA"
        scenario.set_lamp(energy=rnd_energy, color=color, lamp_type=lamp_type)
        return
        
    def validate(self, v, low_b, up_b):
       if v>up_b or v<low_b:
           return False
       return True

    def create_samples(self):
        pose_data = self.get_pose_data()
        used_poses = self.discovery_used_poses()
        for idx, row in  enumerate(pose_data):
            if idx%self.subsampling != 0: #subsampling
                continue
            frame = int(row[0])
            identifier = self.pose_who+"_"+str(frame)
            if identifier in used_poses:
                continue
            file_name_prefix = os.path.join(self.dst,os.path.basename(self.pose_who)+"_"+str(int(frame)))
            if self.constain_face>0:
                if not (self.validate(row[3], -self.constain_face, self.constain_face) and 
                self.validate(-row[2], -self.constain_face, self.constain_face) and self.validate(row[1], -self.constain_face, self.constain_face)):
                    print("face varies too much: (", row[3], -row[2], row[1], "), skip.")
                    continue
            cfg_to_person = (row[3], -row[2], row[1])
            if not (os.path.isfile(file_name_prefix+".png") and os.path.isfile(file_name_prefix+".npy")):
                self.customize_sample(cfg_to_person=cfg_to_person, dst=file_name_prefix, background=self.background_dir)
            print("generate img: ---", os.path.basename(file_name_prefix), "---")
            print("Closing blender")
            p = np.load(file_name_prefix+".npy", allow_pickle=True)
            print(p.item()["config"]["head_pose"], p.item()["config"]["label"])
            bpy.ops.wm.quit_blender()

    def create_samples_allParams(self, imgPath, num=1):
        x = os.path.basename(imgPath).split("_")
        for i in range(0, len(x)):
            x[i] = float(x[i])
        pose_data = self.get_pose_data()
        used_poses = self.discovery_used_poses()
        counter = 0
        while counter < num:
            row = pose_data[randint(0, len(pose_data) - 1)]
            frame = int(row[0])
            identifier = self.pose_who + "_" + str(frame)
            if identifier in used_poses:
                continue
            file_name_prefix = imgPath
            if not (os.path.isfile(file_name_prefix+".png") and os.path.isfile(file_name_prefix+".npy")):
                person = None
                cfg_to_person = (x[0], x[1], x[2])
                if self.model_type == "mhx2":
                    person = IEEPerson(self.model_file)
                    # x=0
                elif self.model_type == "obj":
                    person = IEE4DFacePerson(self.model_file)
                elif self.model_type == "blend":
                    person = IEEMbLabPerson(self.model_file)
                self.hook_ieeperson(person, cfg_to_person)
                scenario = IEEScenario(person=person, res_x=self.img_width, res_y=self.img_height, render_scale=1.0,
                                       label_data=self.coods_3d, GPUs=self.GPUs, background_dir=None)
                #self.hook_ieescenario_params(scenario, [lamp_eng, lamp_col, lamp_loc, lamp_dir, cam_ht])
                self.hook_ieescenario_params(scenario, [x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]])
                scenario.save_3dmodel_to_2dsample(self.coods_3d, file_name_prefix)
                person.clean()
            print("generate img: ---", os.path.basename(file_name_prefix), "---")
            print("Closing blender")
            p = np.load(file_name_prefix+".npy", allow_pickle=True)
            print(p.item()["config"]["head_pose"], p.item()["config"]["label"])
            bpy.ops.wm.quit_blender()
            counter += 1

class IEEImitatorHandonFace(IEEImitator):
    def __init__(self, model_file, label_file, pose_file, img_width, img_height, dst_datafolder=None, sampling=False, constain_face=0, GPUs=[], background_dir=None, subsampling=1):
        super(IEEImitatorHandonFace,self).__init__(model_file, label_file, pose_file, img_width, img_height, dst_datafolder, sampling, constain_face, GPUs, background_dir, subsampling)

    def hook_ieeperson(self, person, cfg_to_person):
        x_ang, y_ang, z_ang = cfg_to_person
        person.adjust_arm() #random ajust the hand pose on face
        person.adjust_face(x_ang, y_ang, z_ang)
        return

class IEEImitatorRndEnv(IEEImitator):
    def __init__(self, model_file, label_file, pose_file, img_width, img_height, dst_datafolder=None, sampling=False, constain_face=0, GPUs=[], background_dir=None):
        super(IEEImitatorRndEnv,self).__init__(model_file, label_file, pose_file, img_width, img_height, dst_datafolder, sampling, constain_face, GPUs, background_dir)

    def hook_ieeperson(self, person, cfg_to_person):
        x_ang, y_ang, z_ang = cfg_to_person
        hand_rnd = randint(0,1)
        if hand_rnd < 0.3:
            person.adjust_arm() #random ajust the hand pose on face
        person.adjust_face(x_ang, y_ang, z_ang)
        return

    def hook_ieescenario(self, scenario, cfg_to_scenario):
        scenario.setup()
        rnd_energy = uniform(0.5,1.5)
        rnd_color = randint(0,2)
        color = (1.0, 1.0, 1.0)
        if rnd_color == 0:
            color = (1.0, 0, 0)
        elif rnd_color == 1:
            color = (0, 1.0, 0)
        else:
            color = (0, 0, 1.0)
        rnd_lamp = randint(0,1)
        lamp_type = "SUN"
        if rnd_lamp == 0:
            lamp_type = "AREA"
        scenario.set_lamp(energy=rnd_energy, color=color, lamp_type=lamp_type)
        return



