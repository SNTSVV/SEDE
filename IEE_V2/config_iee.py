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
from pathlib import Path


#model_type opitons: ".blend", ".obj", ".mhx2"
model_type = ".blend" # For 4d-face put .obj here
constain_face=45
muti_processor = False
GPUs = [0] #ids for GPU to use. GPUs=[] means disable GPU

# org_type options: 1: 3dmodel files and labels are seperated located; 2: 3dmodel file and its label are in one folder
org_type = 2 # For 4dface put 1 here

if model_type==".blend":

    model_folder = "/Users/hazem.fahmy/Documents/HPD/TR/Pool/"
    print("model_folder is set to ", model_folder, "in config_iee.py file, please change to the output path needed accordingly")
    label_folder = None
elif model_type==".obj":
    model_folder = "./data/new4dfacemodels"
    label_folder = "./data/new4dfacelabels"
elif model_type==".mhx2":
    model_folder = "./IEEPackage/mhx2"
    label_folder = "./IEEPackage/newlabel3d"
else:
    model_folder = None
    label_folder = None

pose_folder = "./IEEPackage/pose"
#width = 1920 #752//2 #376
width = 376 #752//2 #376
#height = 1080 #480//2 #240
height = 240 #480//2 #240
#activate_occlusion = True
log_file = "./IEE_V2/logs.log"

subsampling = 1

# use this if you want no backgrounds
background_dir = None
# use this if you want backgrounds
#background_dir = "C:/Users/hazem.fahmy/Documents/fabrizio/snt_simulator/data/hdr_background"

"""
sample type:
1: basic #4dface only support type 1
2: put a hand on face, currently, only support model_type==".mhx2"
3: basic/hand on face + different color/energy
4: not supported yet
"""
sample_type = 1 #now only tested on 4dface, makehuman model may have some bugs
#if sample_type==4: activate_occlusion = True

sample_type_string = {1:"basic_test", 2:"hand_on_face", 3:"rnd_env", 4:"ocllusion"}
#dst_datafolder = "./iee_4dface/"+sample_type_string[sample_type]+"/"
#dst_datafolder = "C:/Users/hazem.fahmy/Documents/HPD/TR/Pool/"+model_type.split(".")[-1] +"/"+sample_type_string[sample_type]+"/"


