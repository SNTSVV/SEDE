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
import glob
import os
import cv2
import dlib
import numpy as np
import pandas as pd
#from shutil import copyfile
from tqdm import tqdm
#import copy
import shutil
#from torch.autograd import Variable
from os.path import join, exists
#import math

import dataSupplier


def ensure_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return


class IEEDataVendor(object):
    def __init__(self, syth_fold_path, kagl_fold_path, face_detector_path, syth_tol=(25,25), real_tol=(10,10), img_size=128):
        print(face_detector_path)
        print("face", os.path.isfile(face_detector_path))
        self.face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        self.syth_fold_path = syth_fold_path
        self.kagl_fold_path = kagl_fold_path
        self.img_size = img_size #the output image size
        self.iee_labels = self.get_iee_labels()
        self.syth_wtol, self.syth_htol = syth_tol
        self.real_wtol, self.real_htol = real_tol


    def get_iee_labels(self):
        labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38,39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70,49,52,55,58]
        return labels

    def get_imgs(self, folder, ext="png"):
        folder_f = folder+"/*."+ext
        f_list = glob.glob(folder_f)
        return f_list

    def save_data(self, data, label, config, dst, file_name):
        ensure_folder(dst)
        dataset = {"data":data, "label":label, "config": config}
        np.save(dst+"/"+file_name+".npy", dataset)
        return

    def split_data(self, data, label, dst, file_name, ratio=0.7):
        ensure_folder(dst)
        r_idx = np.random.permutation(data.shape[0])
        data = data[r_idx]
        label = label[r_idx]

        part_len = int(data.shape[0]*ratio)

        dataset = {"data":data[:part_len], "label":label[:part_len]}
        np.save(dst+"/"+file_name+"_train.npy", dataset)

        dataset = {"data":data[part_len:], "label":label[part_len:]}
        np.save(dst+"/"+file_name+"_test.npy", dataset)

        return


    def get_data_list(self, folder, shuffle=True):
        print(folder)
        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir()]
        if shuffle:
            np.random.shuffle(subfolders)
        return subfolders

    def get_id2kaggle_map(self):
        kag_id_map = {
        70:"left_eye_center",
        69:"right_eye_center",
        43:"left_eye_inner_corner",
        46:"left_eye_outer_corner",
        40:"right_eye_inner_corner",
        37:"right_eye_outer_corner",
        23:"left_eyebrow_inner_end",
        27:"left_eyebrow_outer_end",
        22:"right_eyebrow_inner_end",
        18:"right_eyebrow_outer_end",
        34:"nose_tip",
        55:"mouth_left_corner",
        49:"mouth_right_corner",
        52:"mouth_center_top_lip",
        58:"mouth_center_bottom_lip"
        }
        return kag_id_map

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_kaggle_labels(self, kag_df):
        id2kag = self.get_id2kaggle_map()
        (row, col) = kag_df.shape
        label_arr = np.zeros((row, len(self.iee_labels), 2))

        for idx, ky in enumerate(self.iee_labels):
            if ky in id2kag:
                ky_name = id2kag[ky]
                coords = kag_df[[ky_name+"_x", ky_name+"_y"]].values
                label_arr[:,idx,:]=coords
        return label_arr

    def get_kaggle_data(self):
        df = pd.read_csv(self.kagl_fold_path)
        df = df.fillna(-1)
        df['Image'] = df['Image'].apply(lambda img:  np.fromstring(img, sep = ' '))
        x_data = np.vstack(df['Image'].values)
        x_data = x_data.astype(np.uint8)
        #x_data = x_data / 255.   # scale pixel values to [0, 1]
        x_data = x_data.reshape(-1, 1, 96, 96) # return each images as 1 x 96 x 96
        y_data = self.get_kaggle_labels(df)
        #print("kaggle y_data: ", y_data.shape)
        return x_data, y_data

    # get_real_data wraps get_kaggle_data, and converts the data size to (1,128,128)
    def get_real_data(self):
        x_data, y_data = self.get_kaggle_data()
        new_x_data = []
        new_y_data = []
        noface_num = 0
        for idx in tqdm(range(x_data.shape[0])):
            img_data = x_data[idx,:]
            img_data = cv2.cvtColor(img_data[0,:], cv2.COLOR_GRAY2BGR)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY) #FIXME

            label_data = y_data[idx,:]
            #est biggest face
            img_data = self.get_the_biggest_face(img_data, label_data, self.real_wtol, self.real_htol)
            if not img_data:
                noface_num += 1
                #print("cannot find face in img: ", idx, " skip...", noface_num)
                continue
            new_data, new_label = self.resize(img_data[0], img_data[1], self.img_size)
            new_y_data.append(new_label)
            new_x_data.append(new_data)
        print("INFO: ", noface_num, " imgs cannot find face at kaggle dataset")

        return (np.array(new_x_data), np.array(new_y_data))


    def get_syth_label(self, apng, img_height):
        label_file = apng.split(".png")[0]+".npy"
        #config_file = join(os.path.dirname(apng), "config.npy")
        if not os.path.isfile(label_file):
            return None
        data = np.load(label_file, allow_pickle=True)
        #config_data = np.load(config_file, allow_pickle=True)
        label_data = data.item()["label"]
        #config_data = data.item()["config"]
        #print("label_data: ",label_data)
        label_arr = []
        for ky in self.iee_labels:
            if ky in label_data:
                if label_data[ky][0]  > 0 and label_data[ky][1] > 0:
                    coord = [label_data[ky][0], img_height-label_data[ky][1]]
                    label_arr.append(coord)
                else:
                    label_arr.append([-1, -1]) #(-1,-1) means the keypoint is invisible
            else:
                label_arr.append([0,0]) # label does not exist
        #data.item()["config"]["MBLab"] = config_data.item()
        #data.item()["config"]["MBLab"] = config_data
        #data.item()["config"]["HPD"] = getLabel_V(data.item()["config"]["head_pose"][0]) \
        #                               + getLabel_H(data.item()["config"]["head_pose"][1])
        print(np.array(label_arr))
        return np.array(label_arr), data.item()["config"]

    def get_the_biggest_face(self, img_data, img_label, w_tol, h_tol):
        #print(img_data)
        faces = self.face_detector(img_data,1)
        #print(faces)
        #print(len(faces))
        if len(faces) < 1:
            return None

        big_face = -np.inf
        mx, my, mw, mh = 0, 0, 0, 0
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            if w*h > big_face:
                big_face = w*h
                mx, my, mw, mh = x,y,w,h

        sw_0 = max(mx-w_tol//2,0)
        sw_1 = min(mx+mw+w_tol//2, img_data.shape[1]) #empirical

        sh_0 = max(my-h_tol//2,0)
        sh_1 = min(my+mh+h_tol//2, img_data.shape[0]) #empirical

        assert sh_1 > sh_0
        assert sw_1 > sw_0

        big_face = img_data[sh_0:sh_1, sw_0:sw_1]

        new_label = np.zeros_like(img_label)
        new_label[:,0] = img_label[:,0]-sw_0
        new_label[:,1] = img_label[:,1]-sh_0

        new_label[new_label<0] = 0
        new_label[img_label[:,0]==-1,0] = -1
        new_label[img_label[:,1]==-1,1] = -1 #FIXme : new_label[img_label[:,0]==-1] = -1

        return (big_face, new_label)

    def resize(self, img_face, img_label, target_size):
        (ori_w, ori_h) = img_face.shape
        #print("resize origin: ", img_face.shape, img_label.shape)

        new_img = cv2.resize(img_face, (target_size,target_size), interpolation=cv2.INTER_CUBIC)
        width_resc = float(target_size)/ori_w
        height_resc = float(target_size)/ori_h

        new_label = np.zeros_like(img_label)

        new_label[:,0] = img_label[:,0]*width_resc
        new_label[:,1] = img_label[:,1]*height_resc
        return new_img, new_label

    #@folder_list: must be a list
    def get_syth_data(self, folder_list):
        x_data = []
        y_data = []
        z_data = []
        noface_num = 0
        total_num = 0
        for onefolder in tqdm(folder_list):
            all_pngs = self.get_imgs(onefolder)
            print("Processing ", len(all_pngs), " images")
            for apng in all_pngs:
                print(str((total_num / len(all_pngs) * 100))[0:6] + "%", os.path.basename(apng), end="\r")
                print(os.path.basename(apng))
                img = self.load_img(apng)
                label_data, config_data = self.get_syth_label(apng, img.shape[0])
                big_face = self.get_the_biggest_face(img, label_data, self.syth_wtol, self.syth_htol)

                if not big_face:
                    noface_num += 1
                    #print("cannot find face in img: ", apng, " skip...", noface_num)
                    continue

                new_data, new_label = self.resize(big_face[0], big_face[1], self.img_size)
                y_data.append(new_label)
                x_data.append(new_data)
                z_data.append(config_data)
                total_num += 1

            print("INFO: ", noface_num, " imgs cannot find face at: ", onefolder)

        return (np.array(x_data), np.array(y_data), np.array(z_data))

    #@folder_list: must be a list
    def get_syth_data_inner(self, folder_list, train_ratio=0.8):
        x_data = []
        y_data = []
        print(folder_list)
        for onefolder in tqdm(folder_list):
            noface_num = 0
            all_pngs = self.get_imgs(onefolder)
            for apng in all_pngs:
                img = self.load_img(apng)
                label_data = self.get_syth_label(apng, img.shape[0])
                big_face = self.get_the_biggest_face(img, label_data, self.syth_wtol, self.syth_htol)

                if not big_face:
                    noface_num += 1
                    #print("cannot find face in img: ", apng, " skip...", noface_num)
                    continue

                new_data, new_label = self.resize(big_face[0], big_face[1], self.img_size)
                y_data.append(new_label)
                x_data.append(new_data)

            print("INFO: ", noface_num, " imgs cannot find face at: ", onefolder)

        idata = np.array(x_data)
        ilabel = np.array(y_data)
        idxes = np.array(range(idata.shape[0]))
        np.random.shuffle(idxes)

        train_len = int(len(idxes)*train_ratio)

        train_data = idata[idxes[:train_len]]
        train_label = ilabel[idxes[:train_len]]
        test_data = idata[idxes[train_len:]]
        test_label = ilabel[idxes[train_len:]]

        return (train_data, train_label), (test_data, test_label)

    #generate:train data, test_data, real_data
    def generate_data(self, version= 1, shuffle=True):
        data_list = self.get_data_list(self.syth_fold_path, shuffle=shuffle)
        if version == 1:
            test_set = self.get_syth_data([data_list[0]]) #the input should be a list
            print("---test_set----",test_set[0].shape, test_set[1].shape, test_set[2].shape)
            train_set = self.get_syth_data(data_list[1:])
            print("---train_set----",train_set[0].shape, train_set[1].shape, train_set[2].shape)
        else:
            train_set, test_set = self.get_syth_data_inner(data_list)
            print("train shape: ", train_set[0].shape, "test shape: ", test_set[0].shape)
        #real_set = self.get_real_data()
        real_set = None
        #print("----real_set---",real_set[0].shape, real_set[1].shape)

        return (train_set, test_set, real_set)
    def generate_data_improve(self, version= 2, shuffle=True):
        data_list = self.get_data_list(self.syth_fold_path, shuffle=shuffle)
        if version == 1:
            test_set = self.get_syth_data([data_list[0]]) #the input should be a list
            print("---test_set----",test_set[0].shape, test_set[1].shape, test_set[2].shape)
            train_set = self.get_syth_data(data_list[1:])
            print("---train_set----",train_set[0].shape, train_set[1].shape, train_set[2].shape)
        else:
            improve_set, test_set = self.get_syth_data_inner(data_list, 1)
            print("train shape: ", improve_set[0].shape)
        #real_set = self.get_real_data()
        real_set = None
        #print("----real_set---",real_set[0].shape, real_set[1].shape)

        return (train_set, test_set, real_set)

    def save_evidence(self, imgs, labels, dst, red_factor=20):
        assert imgs.shape[0] == labels.shape[0]

        def update(img, x_t, y_t):
            (height, width, c) = img.shape
            for idx in [-1,0,1]:
                tx = max(min(x_t+idx, width-1),0)
                for jdx in [-1,0,1]:
                    ty = max(min(y_t+jdx, height-1),0)
                    if width>ty > 0 and height>tx >0:
                        img[ty,tx, 0] = 0
                        img[ty,tx, 1] = 0
                        img[ty,tx, 2] = 255
            return img

        for idx in tqdm(range(imgs.shape[0]//red_factor)):
            img = imgs[idx,:]
            img = np.repeat(img[ :, :, np.newaxis], 3, axis=2)
            label = labels[idx,:]
            for jdx in range(label.shape[0]):
                x_t = int(label[jdx][0]+0.5)
                y_t = int(label[jdx][1]+0.5)
                if x_t>0 and y_t>0:
                    img = update(img, x_t, y_t)

            file_name = dst+"/"+str(idx)+".png"
            ensure_folder(dst)
            cv2.imwrite(file_name,img)

def labelHPDimages(npPath, dataSet, originalDst, prefix, mainCounter):
        x1 = np.load(npPath, allow_pickle=True)
        #print(x.item())
        x = x1.item()["config"]
        if mainCounter is None:
            mainCounter = 0
        if not exists(originalDst):
            os.mkdir(originalDst)
        else:
            shutil.rmtree(originalDst)
            os.mkdir(originalDst)
        for img in x1.item()["data"]:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            HP1 = x[mainCounter]["head_pose"][0]
            HP2 = x[mainCounter]["head_pose"][1]
            HP3 = x[mainCounter]["head_pose"][2]
            L1 = getLabel_V(HP1)
            L2 = getLabel_H(HP2)
            label = L1 + L2
            print(mainCounter, HP1, HP2, HP3, label)
            file_name = join(originalDst, label, prefix + str(mainCounter) + ".png")
            if not exists(join(originalDst, label)):
                os.mkdir(join(originalDst, label))
            cv2.imwrite(file_name, img)
            mainCounter += 1

def getLabel_V(HP1):
    if HP1 > 10:  # looking down
        L1 = "Bottom"
    elif HP1 < -10:  # looking up
        L1 = "Top"
    else:
        L1 = "Middle"
    #if HP3 > 35 or HP3 < -35:
    #    L1 = "Undefined"

    return L1
def getLabel_H(HP2):
    if HP2 > 10:
        L2 = "Right"
    elif HP2 < -10:
        L2 = "Left"
    else:
        L2 = "Center"
        #if HP3 < -20:
        #    L2 = "Left"
    return L2

def generate_data(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder, data_version):
    vendor = IEEDataVendor(syth_fold_path, kaggle_fold_path, face_detector_path)
    (train_set, test_set, real_set) = vendor.generate_data(version=data_version, shuffle=True)
    vendor.save_evidence(test_set[0], test_set[1], dst_folder+"/test_env")
    #vendor.save_evidence(real_set[0], real_set[1], dst_folder+"/real_env")
    vendor.save_evidence(train_set[0], train_set[1], dst_folder+"/train_env")
    vendor.save_data(test_set[0], test_set[1], test_set[2], dst_folder, "ieetest")
    #vendor.save_data(real_set[0], real_set[1], dst_folder, "ieereal")
    vendor.save_data(train_set[0], train_set[1], train_set[2], dst_folder, "ieetrain")
    return

def generate_data_improve(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder, data_version):
    vendor = IEEDataVendor(syth_fold_path, kaggle_fold_path, face_detector_path)
    (train_set, test_set, real_set) = vendor.generate_data_improve(version=data_version, shuffle=True)
    vendor.save_evidence(train_set[0], train_set[1], dst_folder+"/improve_env")
    vendor.save_data(train_set[0], train_set[1], train_set[2], dst_folder, "ieeimprove")
    return


def load_data(data_path):
        data_supplier = dataSupplier.DataSupplier(data_path, 64, True, True, 0)
        return data_supplier.get_data_iters()


if __name__ == '__main__':
    path = "C:/Users/hazem.fahmy/Documents/HPD"
    dst_folder = join(path,"test")
    HPD_train = join(path,"TrainingSet")
    HPD_test = join(path,"TestSet")
    HPD_improve = join(path,"ImprovementSet")
    #syth_fold_path = "./frontal_face/basic_mh"
    syth_fold_path = join(path,"./data/output/iee_dataset/blend/basic_test/mblab/")
    syth_fold_path = join(path,"TR/Pool_")
    #syth_fold_path = "./data/mblab"
    kaggle_fold_path = join(path,"IEEPackage/kaggledata/training.csv")
    face_detector_path = join(path,"IEEPackage/clsdata/mmod_human_face_detector.dat")
    data_version = 2 #1. leave one folder as the test data #2. merge data from all folders, keep train_ratio as the training data
    #npPath1 = join(dst_folder, "ieetrain.npy")
    #npPath2 = join(dst_folder, "ieetest.npy")
    npPath = join(dst_folder, "ieeimprove.npy")

    #if not (os.path.isfile(npPath1) and os.path.isfile(npPath2)):
    #    generate_data(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder, data_version)
    generate_data_improve(syth_fold_path, kaggle_fold_path, face_detector_path, dst_folder, data_version)

    simDataSet, _ = load_data(npPath)
    labelHPDimages(npPath, simDataSet, HPD_improve, "", 0)

    #simDataSet, _ = load_data(npPath)
    #labelHPDimages(npPath, simDataSet, HPD_test, "", 0)
