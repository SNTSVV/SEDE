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

import os
import sys
from os.path import basename
import math
import bpy
import bpy_extras
import numpy as np
import mathutils
import bmesh
from mathutils.bvhtree import BVHTree
#from bpy.app.handlers import persistent
from mathutils import Euler
#import utils2
import logging
import IEE_V2.config_iee as cfg
from pathlib import Path
import random
logging.basicConfig(filename=cfg.log_file,level=logging.DEBUG)

def print_info(msg):
    text_file = open("info.txt", "a")
    text_file.write("INFO: %s \n" % msg)
    text_file.close()

class IEEArmPose(object):
    def __init__(self, bones, model_type='mhx'):
        bpy.ops.object.mode_set(mode='POSE')
        self.bones = bones
        self.model_type = model_type

    def pose_left_arm_1_mblab(self):
        l_a_l = self.bones['lowerarm_L']
        u_a_l = self.bones['upperarm_L']
        clavicle_l = self.bones['clavicle_L']

        clavicle_l.rotation_mode = "XYZ"
        l_a_l.rotation_mode = "XYZ"
        u_a_l.rotation_mode = "XYZ"

        clavicle_l.rotation_euler[0] = math.radians(0) 
        clavicle_l.rotation_euler[1] = math.radians(0)
        clavicle_l.rotation_euler[2] = math.radians(0)

        u_a_l.rotation_euler[0] = math.radians(59.) 
        u_a_l.rotation_euler[1] = math.radians(38.4)
        u_a_l.rotation_euler[2] = math.radians(16.4)

        l_a_l.rotation_euler[0] = math.radians(149.)
        l_a_l.rotation_euler[1] = math.radians(-7.05)
        l_a_l.rotation_euler[2] = math.radians(-76.9)
        return

    def pose_left_arm_1_mhx(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]


        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"

        axis = 'X'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=95
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_left_arm_1(self):
        if self.model_type == 'mhx':
            self.pose_left_arm_1_mhx()
        elif self.model_type == 'mblab':
            self.pose_left_arm_1_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")

    def pose_left_arm_2_mhx(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]
        lal02 = self.bones["lowerarm02.L"]
        shoulder = self.bones["shoulder01.L"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"

        #--shoulder--
        axis = 'X'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=40
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=40
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=85
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        return

    def pose_left_arm_2_mblab(self):
        l_a_l = self.bones['lowerarm_L']
        u_a_l = self.bones['upperarm_L']
        clavicle_l = self.bones['clavicle_L']

        clavicle_l.rotation_mode = "XYZ"
        l_a_l.rotation_mode = "XYZ"
        u_a_l.rotation_mode = "XYZ"

        clavicle_l.rotation_euler[0] = math.radians(0) 
        clavicle_l.rotation_euler[1] = math.radians(0)
        clavicle_l.rotation_euler[2] = math.radians(0)

        u_a_l.rotation_euler[0] = math.radians(14.2) 
        u_a_l.rotation_euler[1] = math.radians(28.9)
        u_a_l.rotation_euler[2] = math.radians(7.24)

        l_a_l.rotation_euler[0] = math.radians(122.)
        l_a_l.rotation_euler[1] = math.radians(46.6)
        l_a_l.rotation_euler[2] = math.radians(-89.1)
        return

    def pose_left_arm_2(self):
        if self.model_type == 'mhx':
            self.pose_left_arm_2_mhx()
        elif self.model_type == 'mblab':
            self.pose_left_arm_1_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")


    def pose_left_arm_3_mhx(self):
        ua101 = self.bones["upperarm01.L"]
        ua102 = self.bones["upperarm02.L"]
        lal01 = self.bones["lowerarm01.L"]
        lal02 = self.bones["lowerarm02.L"]
        shoulder = self.bones["shoulder01.L"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"
        #--shoulder--
        axis = 'X'
        angle=10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=55+45
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=100
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=30
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=30
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=-50
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        return

    def pose_left_arm_3_mblab(self):
        l_a_l = self.bones['lowerarm_L']
        u_a_l = self.bones['upperarm_L']
        clavicle_l = self.bones['clavicle_L']

        clavicle_l.rotation_mode = "XYZ"
        l_a_l.rotation_mode = "XYZ"
        u_a_l.rotation_mode = "XYZ"

        clavicle_l.rotation_euler[0] = math.radians(0) 
        clavicle_l.rotation_euler[1] = math.radians(0)
        clavicle_l.rotation_euler[2] = math.radians(0)

        u_a_l.rotation_euler[0] = math.radians(49.9) 
        u_a_l.rotation_euler[1] = math.radians(29.4)
        u_a_l.rotation_euler[2] = math.radians(24.6)

        l_a_l.rotation_euler[0] = math.radians(140)
        l_a_l.rotation_euler[1] = math.radians(35)
        l_a_l.rotation_euler[2] = math.radians(-91.2)
        return

    def pose_left_arm_3(self):
        if self.model_type == 'mhx':
            self.pose_left_arm_3_mhx()
        elif self.model_type == 'mblab':
            self.pose_left_arm_3_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")


    def pose_right_arm_1_mblab(self):
        l_a_r = self.bones['lowerarm_R']
        u_a_r = self.bones['upperarm_R']

        l_a_r.rotation_mode = "XYZ"
        u_a_r.rotation_mode = "XYZ"

        u_a_r.rotation_euler[0] = math.radians(32.9) 
        u_a_r.rotation_euler[1] = math.radians(-17.6)
        u_a_r.rotation_euler[2] = math.radians(-32.4)

        l_a_r.rotation_euler[0] = math.radians(140.)
        l_a_r.rotation_euler[1] = math.radians(30.5)
        l_a_r.rotation_euler[2] = math.radians(29.5)
        return

    def pose_right_arm_1_mhx(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]


        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"

        axis = 'X'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=95
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        return
        return

    def pose_right_arm_1(self):
        if self.model_type == 'mhx':
            self.pose_right_arm_1_mhx()
        elif self.model_type == 'mblab':
            self.pose_right_arm_1_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")


    def pose_right_arm_2_mhx(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]
        lal02 = self.bones["lowerarm02.R"]
        shoulder = self.bones["shoulder01.R"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"

        #--shoulder--
        axis = 'X'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=40
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=40
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=-10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=10
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=85
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_right_arm_2_mblab(self):
        l_a_r = self.bones['lowerarm_R']
        u_a_r = self.bones['upperarm_R']

        l_a_r.rotation_mode = "XYZ"
        u_a_r.rotation_mode = "XYZ"

        u_a_r.rotation_euler[0] = math.radians(114.) 
        u_a_r.rotation_euler[1] = math.radians(-2.5)
        u_a_r.rotation_euler[2] = math.radians(-49.9)

        l_a_r.rotation_euler[0] = math.radians(175.)
        l_a_r.rotation_euler[1] = math.radians(10.1)
        l_a_r.rotation_euler[2] = math.radians(100.)
        return

    def pose_right_arm_2(self):
        if self.model_type == 'mhx':
            self.pose_right_arm_2_mhx()
        elif self.model_type == 'mblab':
            self.pose_right_arm_2_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")
    

    def pose_right_arm_3_mhx(self):
        ua101 = self.bones["upperarm01.R"]
        ua102 = self.bones["upperarm02.R"]
        lal01 = self.bones["lowerarm01.R"]
        lal02 = self.bones["lowerarm02.R"]
        shoulder = self.bones["shoulder01.R"]

        ua101.rotation_mode = "XYZ"
        ua102.rotation_mode = "XYZ"
        lal01.rotation_mode = "XYZ"
        lal02.rotation_mode = "XYZ"
        shoulder.rotation_mode = "XYZ"
        #--shoulder--
        axis = 'X'
        angle=10
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=0
        shoulder.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--upper arms--
        axis = 'X'
        angle=55+45
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=0
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=30
        ua101.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        ua102.rotation_euler.rotate_axis(axis, math.radians(angle))

        #--lower arms--
        axis = 'X'
        angle=100
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Y'
        angle=30
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=30
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))

        axis = 'Z'
        angle=-50
        lal01.rotation_euler.rotate_axis(axis, math.radians(angle))
        angle=0
        lal02.rotation_euler.rotate_axis(axis, math.radians(angle))
        return

    def pose_right_arm_3_mblab(self):
        l_a_r = self.bones['lowerarm_R']
        u_a_r = self.bones['upperarm_R']
        clavicle_r = self.bones['clavicle_R']

        clavicle_r.rotation_mode = "XYZ"
        l_a_r.rotation_mode = "XYZ"
        u_a_r.rotation_mode = "XYZ"

        clavicle_r.rotation_euler[0] = math.radians(1.11) 
        clavicle_r.rotation_euler[1] = math.radians(-0.287)
        clavicle_r.rotation_euler[2] = math.radians(-7.6)

        u_a_r.rotation_euler[0] = math.radians(67.4) 
        u_a_r.rotation_euler[1] = math.radians(-21.)
        u_a_r.rotation_euler[2] = math.radians(-88.7)

        l_a_r.rotation_euler[0] = math.radians(197.)
        l_a_r.rotation_euler[1] = math.radians(-6.76)
        l_a_r.rotation_euler[2] = math.radians(106.)
        return

    def pose_right_arm_3(self):
        if self.model_type == 'mhx':
            self.pose_right_arm_3_mhx()
        elif self.model_type == 'mblab':
            self.pose_right_arm_3_mblab()
        else:
            raise RuntimeError(f"Unknown model type {self.model_type}")

class IEE4DFacePerson(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.texture_path = filepath.split(".obj")[0]+".isomap.png"
        self.model_type = basename(filepath).split(".")[-1]
        self.object_name = basename(filepath).split(".")[0]
        self.imported_object = self.load_model()
        self.objectx = bpy.data.objects[self.object_name]
        self.head_pose = None
        #self.objectx.location = (0, 0, 0)
        #self.objectx.rotation_euler = (0, 0, 0)
        return
    
    #jerry (jdr) @ iee contributed this function
    # Sources
    # https://blender.stackexchange.com/questions/48584/how-to-add-a-texture-to-a-material-using-python
    # https://blender.stackexchange.com/questions/14097/set-active-image-node-with-python/14115#14115
    def load_model(self):
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        #for scene in bpy.data.scenes:
        #    scene.render.engine = 'CYCLES'

        imported_object = bpy.ops.import_scene.obj(filepath=self.filepath)
        new_img = bpy.data.images.load(filepath = self.texture_path)
        
        mat = bpy.data.materials.new(name="FaceTexture")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(self.texture_path)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        
        # Assign it to object
        # Add new material to object
        ob = bpy.data.objects[self.object_name]
        if ob.data.materials:
            # assign to 1st material slot
            ob.data.materials[0] = mat
        else:
            # no slots
            ob.data.materials.append(mat)
        
        ob.scale = [0.03, 0.03, 0.03]
        
        for area in bpy.context.screen.areas: # iterate through areas in current screen
            if area.type == 'VIEW_3D':
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == 'VIEW_3D': # check if space is a 3D view
                        space.shading.type = 'RENDERED' # set the viewport shading to rendered
        
        return ob
    
    def adjust_face(self, x_ang=0, y_ang=0, z_ang=0):
        """
        Given an object `instance`, rotate it by the given angles.
        The rotation angles are assumed to be provided in radians. 
        This convention can be changed to degrees, the below, the 
        angles need to be wrapped in math.radians.
        """
        person = bpy.data.objects[self.object_name]
        if np.abs(x_ang) > 1e-7:
            axis = 'X'
            person.rotation_euler.rotate_axis(axis, math.radians(x_ang))
        if np.abs(y_ang) > 1e-7:
            axis = 'Y'
            person.rotation_euler.rotate_axis(axis, math.radians(y_ang))
        if np.abs(z_ang) > 1e-7:
            axis = 'Z'
            person.rotation_euler.rotate_axis(axis, math.radians(z_ang))
        self.head_pose = (x_ang, y_ang, z_ang)

    def adjust_neck(self, ang, axis="X"):
        return None
       
    def get_nose_focus(self, shift=(0,0,0)):
        return None

    def get_breast_focus(self, shift=(0,0,0)):
        return None

    def get_forehead_focus(self, shift=(0,0,0)):
        return None

    def clean(self):
        bpy.ops.wm.read_homefile()
        return
        
class IEEMbLabPerson(object):
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.model_type = basename(filepath).split(".")[-1]
        self.imported_object = self.load_model()
        self.object_name = self.filepath.stem+"_col"
        self.objectx = bpy.data.objects[self.object_name]
        self.object_mesh_only = bpy.data.objects[self.filepath.stem+"_obj"]
        self.bones = self.objectx.pose.bones

    #jerry (jdr) @ iee contributed this function
    def load_model(self):
        bpy.ops.wm.open_mainfile(filepath=str(self.filepath))
        # Turn off the rig
        try:
            face_rig = bpy.data.objects[f"{self.filepath.stem+'_obj'}_face_rig"]
            bpy.ops.object.select_all(action='DESELECT')
            face_rig.select_set(True)
            bpy.data.objects.remove(face_rig, do_unlink=True)
        except Exception as e:
            # Face rig does not seem to be there. Do nothing
            pass
 
        try:
            phoneme_rig = bpy.data.objects[f"{self.filepath.stem+'_obj'}_phoneme_rig"]
            bpy.ops.object.select_all(action='DESELECT')
            phoneme_rig.select_set(True)
            bpy.data.objects.remove(phoneme_rig, do_unlink=True)
        except Exception as e:
            # Face rig does not seem to be there. Do nothing
            pass

    
    # todo: remove input lamp_type
    def adjust_arm(self, idx=None, lamp_type="SUN"):
        #lamp_type in ["HEMI", "SUN", "POINT", "SPOT", "AREA"]
        self.objectx.select_set(True)
        bpy.context.view_layer.objects.active = self.objectx
        armpose = IEEArmPose(self.bones, model_type='mblab')
        #if lamp_type:
        #    lamp = bpy.data.objects["Lamp"].data
        #    lamp.type = lamp_type

        if not idx:
            idx = np.random.randint(1,7) #if the "_both_" functions are implemented to (1,11)

        if idx == 1:
            armpose.pose_left_arm_1()
        elif idx == 2:
            armpose.pose_left_arm_2()
        elif idx == 3:
            armpose.pose_left_arm_3()
        elif idx == 4:
            armpose.pose_right_arm_1()
        elif idx == 5:
            armpose.pose_right_arm_2()
        elif idx == 6:
            armpose.pose_right_arm_3()
        elif idx == 7:
            armpose.pose_both_arm_1()
        elif idx == 8:
            armpose.pose_both_arm_2()
        elif idx == 9:
            armpose.pose_both_arm_3()

    def load_model_x(self):
        # First clean the scene
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
            
        #2: Use cycles renderer; NOTE: Everything is renderer dependent!
        # What works in cycles might not in Blender Render
        #for scene in bpy.data.scenes:
        #    scene.render.engine = 'CYCLES'
            
        # Load the blend file content
        with bpy.data.libraries.load(str(self.filepath)) as (data_from, _):
            files = []
            for obj in data_from.objects:
                files.append({'name' : obj})
            bpy.ops.wm.append(directory=str(self.filepath)+'\\Object\\', files=files)
        
        # Access the mesh of the human character as well as the whole collection including all of 
        # its assets, e.g. hair, glasses, mask, etc.
        #model_mesh_only = bpy.data.objects[f"{data.stem}_obj"]
        #obj_name = self.filepath.stem+"_col"
        #self.person = bpy.data.objects[obj_name]
        
    def adjust_face(self, x_ang=0, y_ang=0, z_ang=0):
        """
        Given an object `instance`, rotate it by the given angles.
        The rotation angles are assumed to be provided in radians. 
        This convention can be changed to degrees, the below, the 
        angles need to be wrapped in math.radians.
        """
        
        """
        Given an object `instance`, rotate its head by the given angles.

        yaw = z_ang
        pitch = y_ang
        roll = x_ang
        """
        person = bpy.data.objects[self.object_name]
        # Select the human model skeleton and switch to pose mode
        person.select_set(True)
        bpy.context.view_layer.objects.active = person
        bpy.ops.object.mode_set(mode="POSE")

        # Access the head bone
        head = person.pose.bones['head']
        head.rotation_mode = "XYZ"
        head.rotation_euler = [math.radians(x_ang), math.radians(y_ang), math.radians(z_ang)]
        self.head_pose = (x_ang, y_ang, z_ang)
        # Go back to object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        
        """
        if np.abs(x_ang) > 1e-7:
            axis = 'X'
            head.rotation_euler.rotate_axis(axis, math.radians(x_ang))
        if np.abs(y_ang) > 1e-7:
            axis = 'Y'
            head.rotation_euler.rotate_axis(axis, math.radians(y_ang))
        if np.abs(z_ang) > 1e-7:
            axis = 'Z'
            head.rotation_euler.rotate_axis(axis, math.radians(z_ang))
        """

    def adjust_neck(self, ang, axis="X"):
        return None
       
    def get_nose_focus(self, shift=(0,0,0)):
        return None

    def get_breast_focus(self, shift=(0,0,0)):
        return None

    def get_forehead_focus(self, shift=(0,0,0)):
        return None
    
    def clean(self):
        bpy.ops.wm.read_homefile()
        return
        
    def include_face_landmarks_blend(self, labels, col=(1, 0, 0, 1), scale=0.0015):
        """
        Include the facial landmarks.

        Parameters
        ----------
        col: Tuple
            The color of the facial landmark spheres. First three components are the 
            rgb values in [0, 1] and the last value is the alpha value.
        """
        # Create the label material
        mat = bpy.data.materials.new(name="MaterialName")
        mat.diffuse_color = col #change color

        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh_with_modifiers = self.object_mesh_only.evaluated_get(depsgraph)

        for label in labels:
            if label in {'meshname','mask_meshname'}:
                continue
            v = mesh_with_modifiers.data.vertices[labels[label]]
            # Create a sphere
            bpy.ops.mesh.primitive_uv_sphere_add(radius=scale, location=v.co, enter_editmode=False)
            s = bpy.data.objects['Sphere']
            s.name = f'sphere_{label}'
            s.data.materials.append(mat)
            s.parent = self.object_mesh_only
        
#creat a person object
#input: file path of makehuman mhx2 model
class IEEPerson(object):
    def __init__(self,filepath):
        self.filepath=filepath
        self.object_name = basename(filepath).split(".")[0]
        self.model_type = basename(filepath).split(".")[-1]
        self.person_name = self.object_name[:3]
        self.imported_object = self.load_model()

        self.objectx = bpy.data.objects[self.object_name]

        bpy.ops.object.mode_set(mode='POSE')
        print("DATA", bpy.ops.object.pose.bones.keys())

        self.bones = self.objectx.pose.bones

        self.head_pose = None
        return

    def get_nose_focus(self, shift=(0,0,0)):
        bx = self.bones["special03"].tail
        return np.array(bx)+shift

    def get_breast_focus(self, shift=(0,0,0)):
        bx = (self.bones["breast.R"].tail + self.bones["breast.L"].tail)/2.0
        return np.array(bx)+shift

    def get_forehead_focus(self, shift=(0,0,0)):
        bx = (self.bones["oculi01.R"].tail + self.bones["oculi01.L"].tail)/2.0
        return np.array(bx)+shift


    def load_model(self):
        # switch of log information
        #---------------------------------------------------------
        logfile = 'blender_load.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        
        f_ext = self.filepath.split(".")[-1]
     
        #imported_object = bpy.ops.import_scene.makehuman_mhx2(filepath=self.filepath)

        imported_object = bpy.ops.import_scene.obj(filepath=self.filepath)  #Blender 2.81
        os.close(1)
        os.dup(old)
        os.close(old)

        return imported_object

    
    # todo: remove input lamp_type
    def adjust_arm(self, idx=None, lamp_type="SUN"):
        #lamp_type in ["HEMI", "SUN", "POINT", "SPOT", "AREA"]
        armpose = IEEArmPose(self.bones, model_type='mhx')
        #if lamp_type:
        #    lamp = bpy.data.objects["Lamp"].data
        #    lamp.type = lamp_type

        if not idx:
            idx = np.random.randint(1,7) #if the "_both_" functions are implemented to (1,11)

        if idx == 1:
            armpose.pose_left_arm_1()
        elif idx == 2:
            armpose.pose_left_arm_2()
        elif idx == 3:
            armpose.pose_left_arm_3()
        elif idx == 4:
            armpose.pose_right_arm_1()
        elif idx == 5:
            armpose.pose_right_arm_2()
        elif idx == 6:
            armpose.pose_right_arm_3()
        elif idx == 7:
            armpose.pose_both_arm_1()
        elif idx == 8:
            armpose.pose_both_arm_2()
        elif idx == 9:
            armpose.pose_both_arm_3()


    def adjust_face(self, x_ang=0, y_ang=0, z_ang=0):
        # to move the bones, we need to go into pose mode
        bpy.ops.object.mode_set(mode='POSE')
        head = self.bones["head"] # manually selected skeleton
        head.rotation_mode = "XYZ"
        print("transform: ", x_ang, -y_ang, z_ang)
        if np.abs(x_ang) > 1e-7:
            axis = 'X'
            head.rotation_euler.rotate_axis(axis, math.radians(x_ang))
        if np.abs(y_ang) > 1e-7:
            axis = 'Y'
            head.rotation_euler.rotate_axis(axis, math.radians(y_ang))
        if np.abs(z_ang) > 1e-7:
            axis = 'Z'
            head.rotation_euler.rotate_axis(axis, math.radians(z_ang))
        self.head_pose = (x_ang, y_ang, z_ang)
        return

    def adjust_neck(self, ang, axis="X"):
        bpy.ops.object.mode_set(mode='POSE')
        neck1 = self.bones["neck01"]
        neck1.rotation_mode = "XYZ"
        neck2 = self.bones["neck02"]
        neck2.rotation_mode = "XYZ"

        neck1.rotation_euler.rotate_axis(axis, math.radians(ang))
        neck2.rotation_euler.rotate_axis(axis, math.radians(ang))

    
    def set_neck(self, angle_scale=15, random_level=0.2):
        #Empirically moving both neck1 and neck2, looks more natural
        bpy.ops.object.mode_set(mode='POSE')
        neck2 = self.bones["neck02"]
        neck1 = self.bones["neck01"]
        neck2.rotation_mode = "XYZ"
        neck1.rotation_mode = "XYZ"
        
        # rotate X
        axis = 'X'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Y
        
        axis = 'Y'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Z
        axis = 'Z'
        angle = np.random.normal(0,random_level)*angle_scale
        neck1.rotation_euler.rotate_axis(axis, math.radians(angle))
        neck2.rotation_euler.rotate_axis(axis, math.radians(angle))
        return
    
    def set_spine(self, angle_scale=15, random_level=0.15):
        bpy.ops.object.mode_set(mode='POSE')
        spine2 = self.bones["spine02"]
        spine1 = self.bones["spine01"]
        spine2.rotation_mode = "XYZ"
        spine1.rotation_mode = "XYZ"
        
        # rotate X
        axis = 'X'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Y
        
        axis = 'Y'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))
        # rotate Z
        axis = 'Z'
        angle = np.random.normal(0,random_level)*angle_scale
        spine1.rotation_euler.rotate_axis(axis, math.radians(angle))
        spine2.rotation_euler.rotate_axis(axis, math.radians(angle))
         
    def random_transform(self):
        self.set_neck()
        self.set_spine()
        return
        
    def eye_closed(self):
        #eye status was marked in the file_name when generating MH models
        mark = self.object_name.split("_")[1]
        if mark == "o":
            return False
        return True
    
    def clean(self):
        clr_flag = True
        bpy.ops.wm.read_homefile()
        return
    

class IEEScenario(object):
    clr_flag = False

    def __init__(self, person, res_x, res_y, render_scale = 1.0, label_data=None, GPUs=[], background_dir=None):
        if IEEScenario.clr_flag:
            print("WARN: the scenario is not clear!")
        IEEScenario.clr_flag = False

        self.label_data = label_data
        self.background = None
        
        self.res_x = res_x
        self.res_y = res_y
        self.person = person
        
        if person.model_type == "mhx2":
            self.nose_focus = person.get_nose_focus(shift=(0,0,-0.1)) #empirically
            self.forehead_focus = person.get_forehead_focus()
            self.breast_focus = person.get_breast_focus(shift=(0,-4.5,1.5)) #empirically

        self.scene = bpy.context.scene
        if len(GPUs)>0:
            self.enable_gpu(GPUs)
        self.objectx = self.person.objectx

        #self.scene.objects.active = self.objectx
        bpy.context.view_layer.objects.active = self.objectx
        self.mat_world = self.objectx.matrix_world 

        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = 100

        self.render_size = (int(self.res_x * render_scale), int(self.res_y * render_scale))
        self.cam = self.scene.objects['Camera']
        #self.lamp = self.scene.objects['Lamp']
        self.lamps = {"Light":1}

        self.cam_looking_direction = None
        self.background_dir = background_dir

        self.min_x = 1e7
        self.min_y = 1e7
        self.max_x = -1e7
        self.max_y = -1e7
        self.gaze = -1
        return
        
    def enable_gpu(self,GPUs):
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
 
        # get the settings (user preferences)
        prefs = bpy.context.preferences.addons['cycles'].preferences
 
        # specify to use CUDA
        prefs.compute_device_type = 'CUDA'
 
        # check the actual devices installed (the above is only the ones in the preferences)
        cuda_devices, _ = prefs.get_devices()
 
        # do not consider the last one (i.e. the CPU)
        #only use half the GPUs, if more than 1 GPU.
        #cuda_num = 1
        #if len(cuda_devices)>1:
        #    cuda_num = len(cuda_devices)//2
        #for i in range(cuda_num):
        #    cuda_devices[i].use = True
        for gpu_id in range(len(cuda_devices)):
            cuda_devices[gpu_id].use = False
        for gpu_id in GPUs:
            cuda_devices[gpu_id].use = True #hardcode....
 
        # save the settings to the profile
        #bpy.ops.wm.save_userpref()
        
        
    def set_camera(self, location=None, target=None, random_loc=0, random_tgt=0):
        if self.person.model_type == "mhx2":
            self.set_camera_mhx2(location,target,random_loc,random_tgt)
        elif self.person.model_type == "obj":
            self.set_camera_obj(location,target,random_loc,random_tgt)
        elif self.person.model_type == "blend":
            self.set_camera_blend(location,target,random_loc,random_tgt)
        else:
            print("No support of ", self.person.model_type)
        #bpy.context.scene.update()
        bpy.context.view_layer.update()
        self.mat_world = self.objectx.matrix_world 
    
    # JERRY: HERE THE CAMERA IS SET
    def set_camera_blend(self, location=None, target=None, random_loc=0, random_tgt=0):
        ran_loc = [0,0,0]
        ran_tgt = [0,0,0]

        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh = self.person.object_mesh_only.evaluated_get(depsgraph)

        # JERRY: CHECK WHERE LABEL DATA IS COMING FROM
        z_loc = mesh.data.vertices[self.label_data[28]].co.z
        # JERRY: MAYBE INCREASE THIS A BIT
        z_loc = z_loc*1.1
        #z_loc = z_loc*2

        if random_loc:
            ran_loc = (np.random.rand(3)-0.5)*random_loc #an empirical setting
            
        if random_tgt:
            ran_tgt = (np.random.rand(3)-0.5)*random_tgt

        looking_direction = None
        
        if location:
            self.cam.location = location+random_loc 

            if target:
                looking_direction = target+random_tgt
                self.cam.rotation_euler = looking_direction
            else:
               looking_direction = np.array([math.radians(80.7590),math.radians(-0.000051), math.radians(22.292)])+random_tgt
               self.cam.rotation_euler = looking_direction
        else:
            #self.cam.location = np.array([0, 0, 0])

            self.cam.location = (np.array([0, -1, z_loc]) +ran_loc)
            #self.cam.location = (np.array([0.462829, -1.17942, -1]) +ran_loc)
            #self.cam.location = (np.array([0.462829, -1.17942, z_loc]) +ran_loc)
            #self.cam.location = (np.array([random.uniform(0.44,0.48), random.uniform(-1.2, -1.1), z_loc]))
            print("target", target)
            if target:
                looking_direction = np.array([math.radians(80.7590),math.radians(0.0000), math.radians(target[2])])
                self.cam.rotation_euler = looking_direction
            else:
                looking_direction = np.array([math.radians(80.7590),math.radians(0.0000), math.radians(0)])+random_tgt
                #looking_direction = np.array([math.radians(80.7590),math.radians(1.0000), math.radians(22.292)])+random_tgt
                self.cam.rotation_euler = looking_direction

        self.cam_looking_direction = looking_direction
            
        return

    def set_camera_obj(self, location=None, target=None, random_loc=0, random_tgt=0):
        ran_loc = [0,0,0]
        ran_tgt = [0,0,0]
        
        if random_loc:
            ran_loc = (np.random.rand(3)-0.5)*random_loc #an empirical setting
            
        if random_tgt:
            ran_tgt = (np.random.rand(3)-0.5)*random_tgt

        looking_direction = None
        
        
        if location:
            self.cam.location = location+random_loc 

            if target:
                looking_direction = target+random_tgt
                self.cam.rotation_euler = looking_direction
            else:
               looking_direction = np.array([1.88, 0., 0.])+random_tgt
               self.cam.rotation_euler = looking_direction
        else:
            self.cam.location = np.array([0.0, -16.5, -5]) +ran_loc
            if target:
                looking_direction = target+random_tgt
                self.cam.rotation_euler = looking_direction
            else:
                looking_direction = np.array([1.88, 0., 0.])+random_tgt
                self.cam.rotation_euler = looking_direction

        self.cam_looking_direction = looking_direction
            
        return

    # the random factors are introduced to generate faces in difference
    # location: where is the camera, default: self.breast_focus+ran_loc
    # target: where the camera look at, default: self.nose_focus+ran_tgt
    def set_camera_mhx2(self, location=None, target=None, random_loc=0, random_tgt=0):
        ran_loc = (0,0,0)
        ran_tgt = (0,0,0)
        
        if random_loc:
            ran_loc = (np.random.rand(3)-0.5)*random_loc #an empirical setting
            
        if random_tgt:
            ran_tgt = (np.random.rand(3)-0.5)*random_tgt

        looking_direction = None
        if location:
            self.cam.location = location+random_loc 

            if target:
                looking_direction = self.cam.location - mathutils.Vector(target+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()
            else:
                looking_direction = self.cam.location - mathutils.Vector(self.nose_focus+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()
        else:
            self.cam.location = self.breast_focus+ran_loc 
            if target:
                looking_direction = location - mathutils.Vector(target+ran_tgt)
                self.rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()
            else:
                looking_direction = self.cam.location - mathutils.Vector(self.nose_focus+ran_tgt)
                rot_quat = looking_direction.to_track_quat('Z', 'Y')
                self.cam.rotation_euler = rot_quat.to_euler()

        self.cam_looking_direction = looking_direction
            
        return

    def get_lamp_params(self, name):
        if not name in self.lamps:
            return None
        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data

        lamp_cfg = {}

        lamp_cfg["location"] = lamp_obj.location
        lamp_cfg["type"] = lamp_data.type
        lamp_cfg["color"] = lamp_data.color
        lamp_cfg["energy"] = lamp_data.energy
        lamp_cfg["direct_xyz"] = (lamp_obj.rotation_euler.x, lamp_obj.rotation_euler.y, lamp_obj.rotation_euler.z)

        return lamp_cfg

    #if any manipulation, use the following commands
    # lamp_object.select = True
    # self.scene.objects.active = lamp_object
    def add_lamp(self, name, location, energy=1.0, direct_xyz=(0.873,-0.873,0.698), color=(1.0, 1.0, 1.0),type="POINT"):
        # if the lamp is already existed, nothing to add
        if name in self.lamps:
            return None
        lamp_data = bpy.data.lamps.new(name=name, type=type)
        lamp_data.energy = energy
        lamp_data.color = color
        

        lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)
        lamp_object.rotation_euler = Euler(direct_xyz, "XYZ")
        lamp_object.location = location

        self.scene.objects.link(lamp_object)
        self.lamps[name] = lamp_object

        return lamp_object
    #direct_xyz=(0.873,-0.873,0.698)
    def set_lamp(self, name="Light", energy=1.0, direct=None, color=(1.0, 1.0, 1.0), location=None, random_leval=None, lamp_type="SUN"):
        if self.person.model_type == "mhx2":
            self.set_lamp_mhx2(name=name, energy=energy, 
            direct_xyz=(0.873,-0.873,0.698), color=color, location=location, random_leval=random_leval, lamp_type=lamp_type)
        elif self.person.model_type == "obj":
            self.set_lamp_obj(name, energy, (0.873,-0.873,0.698), color, location, random_leval, lamp_type="SUN")
        elif self.person.model_type == "blend":
            self.set_lamp_blend(name, energy, direct, color, location, random_leval, lamp_type="SUN")
        else:
            print("No support of ", self.person.model_type)
            
    # lamp_type: HEMI, POINT, SUN, SPOT, AREA
    def set_lamp_obj(self, name="Light", energy=1.0, direct_xyz=(0.873,-0.873,0.698), color=(1.0, 1.0, 1.0), location=None, random_leval=None, lamp_type="SUN"):
        # if the lamp name is not in the self.lamps, do nothing
        if not name in self.lamps:
            return None

        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data


        ran_loc = (0,0,0)
        if random_leval:
            ran_loc = (np.random.rand(3)-0.5)*random_leval

        if location:
            lamp_obj.location = location + ran_loc
        else:
            shift = np.array((0,-5,0.5)) + np.array(ran_loc) #empirically 
            lamp_obj.location = [0, -7.10, 1.72] + shift

        lamp_obj.rotation_euler = Euler(direct_xyz, "XYZ")

        if lamp_type in {"HEMI","POINT","SUN","SPOT", "AREA"}:
            lamp_data.type = lamp_type

        lamp_data.color = color

        lamp_data.energy = energy

        return
        
     # lamp_type: HEMI, POINT, SUN, SPOT, AREA
    def set_lamp_blend(self, name="Light", energy=1.0, direct_xyz=None, color=(1.0, 1.0, 1.0), location=None, random_leval=None, lamp_type="SUN"):
        # if the lamp name is not in the self.lamps, do nothing
        if not name in self.lamps:
            return None

        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data


        ran_loc = (0,0,0)
        if random_leval:
            ran_loc = (np.random.rand(3)-0.5)*random_leval

        if location:
            #print(location, ran_loc)
            #lamp_obj.location = location + ran_loc
            lamp_obj.location = location
        else:
            shift = np.array((0,-5,0.5)) + np.array(ran_loc) #empirically 
            shift = np.array((0,-5,0.5)) + np.array(np.random.rand(3) - 0.5) #empirically
            lamp_obj.location = [0, -7.10, 1.72] + shift
        if direct_xyz:
            lamp_obj.rotation_euler = Euler(direct_xyz, "XYZ")
        else:
            lamp_obj.rotation_euler = Euler((random.uniform(0.29, 0.87), random.uniform(-5.0, -0.87), random.uniform(0.68, 14.70)), "XYZ")

        if lamp_type in {"HEMI","POINT","SUN","SPOT", "AREA"}:
            lamp_data.type = lamp_type

        lamp_data.color = color

        lamp_data.energy = energy

        return

    # lamp_type: HEMI, POINT, SUN, SPOT, AREA
    def set_lamp_mhx2(self, name="Light", energy=1.0, direct_xyz=(0.873,-0.873,0.698), color=(1.0, 1.0, 1.0), location=None, random_leval=None, lamp_type="SUN"):
        # if the lamp name is not in the self.lamps, do nothing
        if not name in self.lamps:
            return None

        lamp_obj = bpy.data.objects[name]
        lamp_data = lamp_obj.data


        ran_loc = (0,0,0)
        if random_leval:
            ran_loc = (np.random.rand(3)-0.5)*random_leval

        if location:
            lamp_obj.location = location + ran_loc
        else:
            shift = np.array((0,-5,0.5)) + np.array(ran_loc) #empirically
            lamp_obj.location = self.nose_focus + shift

        lamp_obj.rotation_euler = Euler(direct_xyz, "XYZ")


        if lamp_type in {"HEMI","POINT","SUN","SPOT", "AREA"}:
            lamp_data.type = lamp_type

        lamp_data.color = color

        lamp_data.energy = energy

        return
    def label_headpose(self):
        margin1 = 10 #bottom
        margin2 = -10  # top

        margin3 = 10 #right
        margin4 = -10  # left

        L1 = None
        L2 = None
        HP1 = self.person.head_pose[0]
        HP2 = self.person.head_pose[1]
        HP3 = self.person.head_pose[2]

        if HP1 > 10:  # looking down
            L1 = "Bottom"
        elif HP1 < -10:  # looking up
            L1 = "Top"
        else:
            L1 = "Middle"
        # if HP3 > 35 or HP3 < -35:
        #    L1 = "Undefined"

        if HP2 > 10:
                L2 = "Right"
        elif HP2 < -10:
                L2 = "Left"
        else:
                L2 = "Center"
                # if HP3 < -20:
                #    L2 = "Left"

        return L1+L2

    def get_scenario_param(self):
        scenario = {}
        for ky in self.lamps:
            #lamp = bpy.data.lamps[ky]
            lamp_obj = bpy.data.objects[ky]
            lamp_data = lamp_obj.data

            sce_key = "lamp_loc_"+ky
            scenario[sce_key] = list(lamp_obj.location)
            sce_key = "lamp_type_"+ky
            scenario[sce_key] = lamp_data.type
            sce_key = "lamp_energy_"+ky
            scenario[sce_key] = lamp_data.energy
            sce_key = "lamp_color_"+ky
            scenario[sce_key] = list(lamp_data.color)

            sce_key = "lamp_direct_xyz_"+ky
            scenario[sce_key] = [lamp_obj.rotation_euler.x, lamp_obj.rotation_euler.y, lamp_obj.rotation_euler.z]

        scenario["cam_loc"] = list(self.cam.location)
        scenario["cam_look_direction"] = list(self.cam_looking_direction)
        scenario["head_pose"] = list(self.person.head_pose)
        # add MBLab configuration parameters (eye, age, skin, color)
        if self.background:
            scenario["background"] = str(os.path.basename(str(self.background))).split(".")[0]
        scenario["label"] = self.label_headpose()
        return scenario
    
    def random_transform(self):
        self.set_camera()
        self.set_lamp()

    def setup(self, adj_neck=True, adj_deg=5):
        self.set_camera(random_loc=0, random_tgt=0)
        self.set_lamp(random_leval=0)
        if adj_neck:
            self.person.adjust_neck(adj_deg)
        if self.background_dir:
            self.set_background()

    def set_background(self):
        images = [p for p in Path(self.background_dir).iterdir() if p.suffix == '.hdr']
        if len(images) == 0:
            lograise(f"No suitable background images found in directory '{self.background_dir}'", FileNotFoundError)
        image = random.choice(images)
        self.background = image
        C = bpy.context
        world = C.scene.world
        world.use_nodes = True
        enode = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
        enode.image = bpy.data.images.load(str(image))
        bg = world.node_tree.nodes['Background']
        world.node_tree.links.new(enode.outputs['Color'], bg.inputs['Color'])

        tex_coord = world.node_tree.nodes.new('ShaderNodeTexCoord')
        rotation_mapping = world.node_tree.nodes.new('ShaderNodeMapping')
        
        world.node_tree.links.new(tex_coord.outputs['Generated'], rotation_mapping.inputs['Vector'])
        world.node_tree.links.new(rotation_mapping.outputs['Vector'], enode.inputs['Vector'])

        # Rotate randomly
        rotation_mapping = bpy.data.worlds[world.name].node_tree.nodes["Mapping"]
        # rotation_mapping.inputs[2].default_value[0] = math.radians(degrees_x)
        # rotation_mapping.inputs[2].default_value[1] = math.radians(degrees_y)
        rotation_mapping.inputs[2].default_value[2] = math.radians(random.uniform(0, 360))

    def check_visibility(self, vertex_w_co, vertex_c_co, tolerance = 0.02, pinfo=False): 
        # init the pixel values outside the image
        pixel_x = -1
        pixel_y = -1

        # cast a ray from the camera to the vertex
        #from sds@iee
        # TODO 1: i am still not sure if it is 100% correct.
        #         are cam.location and vertex in same coordinate space?
        # TODO 2: i would propose to spend some additional time to check if this is correct

        ray = self.scene.ray_cast(bpy.context.view_layer, self.cam.location, (vertex_w_co - self.cam.location).normalized())

        # If the ray hits something and if this hit is close to the vertex
        # then we assume this vertex is visible
        if ray[0] and (ray[1]-vertex_w_co).magnitude < tolerance:
            # for plotting we need to use the camera coordinates (but scaled with the render size)
            pixel_x = round(vertex_c_co[0] * self.render_size[0])
            pixel_y = round(vertex_c_co[1] * self.render_size[1])
            if pinfo:
                #assert 0==1
                print_info("-2d-"+str(pixel_x)+"__"+str(pixel_y))
        else:
            pixel_x = -round(vertex_c_co[0] * self.render_size[0])
            pixel_y = -round(vertex_c_co[1] * self.render_size[1])

        return pixel_x, pixel_y
        
    def map_coord_from_3d_to_2d_face(self, label_data, coords_2d, chk_visi=True):
        if not "meshname" in label_data:
            return
        
        # The following solution has be constructed based on:
        # https://blender.stackexchange.com/questions/150095/how-to-get-vertex-coordinates-after-modifier-in-python-in-blender-2-80
        depsgraph = bpy.context.evaluated_depsgraph_get()
        tmp_objectx = bpy.data.objects[label_data["meshname"]]
        #tmp_objectx_msk = bpy.data.objects[label_data["mask_meshname"]]
        
        mat_world = tmp_objectx.matrix_world
        bm_o_mesh = bmesh.new()
        bm_o_mesh.from_object(tmp_objectx, depsgraph)
        bm_o_mesh.verts.ensure_lookup_table()
        bm_o = bm_o_mesh.verts
        
        #mat_world_msk = tmp_objectx_msk.matrix_world
        #bm_o_mesh_msk = bmesh.new()
        #bm_o_mesh_msk.from_object(tmp_objectx_msk, depsgraph)
        #bm_o_mesh_msk.verts.ensure_lookup_table()
        #bm_o_msk = bm_o_mesh_msk.verts

        for idx, ky in enumerate(label_data):
            if not isinstance(ky, int):
                continue
            if 90>ky>68 or ky<1:
                continue
            vid = label_data[ky]
            if vid<1:
                coords_2d[ky] = [-1, -1]
                continue
            # get the vertix for this label
            #if ky>99:
            #    vertex_o_co  = bm_o_msk[vid]
            #else:
            vertex_o_co  = bm_o[vid]
            # transform the coordinates of the vertex (object coordinates system) 
            # to coordinates in the world coordinate system
            # needed for intersection
            #if ky>99:
            #    vertex_w_co = mat_world_msk@vertex_o_co.co
            #else:
            vertex_w_co = mat_world@vertex_o_co.co
            # transform the coordinates from world coordinate system to the camera
            # coordinate system
            # needed for plotting
            vertex_c_co = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, vertex_w_co)

            if chk_visi:
                pixel_x, pixel_y = self.check_visibility(vertex_w_co, vertex_c_co, 0.02)
            else:
                pixel_x = round(vertex_c_co[0] * self.render_size[0])
                pixel_y = round(vertex_c_co[1] * self.render_size[1])

            coords_2d[ky] = [pixel_x, pixel_y]
        #bpy.data.meshes.remove(bm_o_mesh)
        tmp_objectx.to_mesh_clear()
        #tmp_objectx_msk.to_mesh_clear()
        return

    def map_coord_from_3d_to_2d_eye(self, label_data, coords_2d, chk_visi=True):
        if not "eyename" in label_data:
            return    

        eye_object = bpy.data.objects[label_data["eyename"]]
        bpy.context.view_layer.objects.active = eye_object
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bm_o_mesh = bmesh.new()
        bm_o_mesh.from_object(eye_object, depsgraph)
        bm_o_mesh.verts.ensure_lookup_table()
        bm_o = bm_o_mesh.verts
        

        for ky in label_data:
            if not isinstance(ky, int):
                continue
            if ky < 69:
                continue
            vid = label_data[ky]
            print_info("-ky-: "+str(ky)+"_vid_"+str(vid))
   
            if vid<1:
                coords_2d[ky] = [-1, -1]
                continue
            # get the vertix for this label
            vertex_o_co  = bm_o[vid]
            # transform the coordinates of the vertex (object coordinates system) 
            # to coordinates in the world coordinate system
            # needed for intersection
            vertex_w_co = self.mat_world@vertex_o_co.co
            vertex_c_co = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, vertex_w_co)
            
            if chk_visi:
                pixel_x, pixel_y = self.check_visibility(vertex_w_co=vertex_w_co, vertex_c_co=vertex_c_co,tolerance=0.05, pinfo=True)
            else:
                pixel_x = round(vertex_c_co[0] * self.render_size[0])
                pixel_y = round(vertex_c_co[1] * self.render_size[1])

            coords_2d[ky] = [pixel_x, pixel_y]
        #bpy.data.meshes.remove(bm_o_mesh)
        eye_object.to_mesh_clear()
        return


    def save_3dmodel_to_2dsample(self, label_data, file_name_prefix, chk_visi=True):
        # switch of log information
        #---------------------------------------------------------

        logfile = 'IEE_V2/blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
        
        coords_2d = {}
        self.map_coord_from_3d_to_2d_face(label_data, coords_2d, chk_visi)
        self.map_coord_from_3d_to_2d_eye(label_data, coords_2d,chk_visi)

        image_name = file_name_prefix+".png"
        label_name = file_name_prefix+".npy"

        data = {}
        data["config"] = self.get_scenario_param()
        data["label"] = coords_2d
        print(image_name)
        print(label_name)
        np.save(label_name, data)



        #self.scene.render.filepath = image_name
        #bpy.data.scenes["Scene"].render.filepath = image_name
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.filepath = image_name
        #bpy.context.object.hide_viewport = True
        #bpy.context.object.hide_render = True
        bpy.ops.render.render(animation=False, write_still=True)

        os.close(1)
        os.dup(old)
        os.close(old)
        return

    def _min_max(self, co_2d):
        pixel_x = round(co_2d[0] * self.render_size[0])
        pixel_y = round(co_2d[1] * self.render_size[1])
        self.min_x = min(self.min_x, pixel_x)
        self.min_y = min(self.min_y, pixel_y)
        self.max_x = max(self.max_x, pixel_x)
        self.max_y = max(self.max_y, pixel_y)

    def _get_face_label(self):
        bpy.ops.object.mode_set(mode='POSE')

        #calculate face bound box
        #bones not counted in: speical06.R, special06.L, speical05.R, special05.L, head
        #mannully ...
        avoid_set = {"special06.R", "special06.L", "special05.R", "special05.L", "head", "jaw", "special04", "tongue00"}
        avoid_tail_set = {"eye.R", "eye.L", "orbicularis03.R", "orbicularis03.L", "orbicularis04.R", "orbicularis04.L"}

        for subbone in self.person.bones["head"].children_recursive:
            bone_name = subbone.name
            if bone_name in avoid_set:
                continue
            co_2d_h = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*subbone.head)
            self._min_max(co_2d_h)
            if bone_name in avoid_tail_set:
                continue
            co_2d_t = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*subbone.tail)
            self._min_max(co_2d_t)
          
        
        bpy.ops.object.mode_set(mode='OBJECT')
            
        if self.max_x < 10 or self.max_y < 10: #skip bad samples
            return (0,0,0,0)
        if self.min_x > self.res_x - 10 or self.min_y > self.res_y - 10:
            return (0,0,0,0)
        
        self.min_x = max(0, self.min_x)
        self.min_y = max(0, self.min_y)
        self.max_x = min(self.res_x, self.max_x)
        self.max_y = min(self.res_x, self.max_y)
        
        area = (self.max_y-self.min_y)*(self.max_x-self.min_x)
        if area < 100:
            return (0,0,0,0)
        return (self.min_x, self.res_y - self.min_y, self.max_x, self.res_y-self.max_y)
        
        
    def _get_gaze(self, render_scale=1.0):
        if self.person.eye_closed():
            return -1

        bpy.ops.object.mode_set(mode='POSE')
        eye_R = self.person.bones["eye.R"]
        eye_L = self.person.bones["eye.L"]

        rh_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_R.head)
        rt_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_R.tail)
        lh_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_L.head)
        lt_co_2d = bpy_extras.object_utils.world_to_camera_view(self.scene, self.cam, self.mat_world*eye_L.tail)
        bpy.ops.object.mode_set(mode='OBJECT')

        rh_pixel_x = round(rh_co_2d[0] * self.render_size[0])
        rh_pixel_y = round(rh_co_2d[1] * self.render_size[1])
        rt_pixel_x = round(rt_co_2d[0] * self.render_size[0])
        rt_pixel_y = round(rt_co_2d[1] * self.render_size[1])

        lh_pixel_x = round(lh_co_2d[0] * self.render_size[0])
        lh_pixel_y = round(lh_co_2d[1] * self.render_size[1])
        lt_pixel_x = round(lt_co_2d[0] * self.render_size[0])
        lt_pixel_y = round(lt_co_2d[1] * self.render_size[1])

        r_gaze = math.atan2(rt_pixel_y-rh_pixel_y,rt_pixel_x-rh_pixel_x)
        l_gaze = math.atan2(lt_pixel_y-lh_pixel_y,lt_pixel_x-lh_pixel_x)
        
        return round(math.degrees((r_gaze+l_gaze)/2.0))
        
    def save_to_sample(self, t_folder="./imgs/"):

        logfile = 'data/blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

        bbox = self._get_face_label()
        gaze = self._get_gaze()
        s_label = str(bbox[0])+"_"+str(bbox[1])+"_"+str(bbox[2])+"_"+str(bbox[3])+"_"+str(gaze)
        fext = self.person.object_name.split(".")
        if not os.path.exists(t_folder):
                os.mkdir(t_folder)

        t_folder = t_folder+self.person.person_name+"/"
        if not os.path.exists(t_folder):
                os.mkdir(t_folder)

        filepath = t_folder+fext[0]+"_"+s_label+".png"
        self.scene.render.filepath = filepath
        bpy.ops.render.render( write_still=True)

        os.close(1)
        os.dup(old)
        os.close(old)
        return
                
        
        
    
    
