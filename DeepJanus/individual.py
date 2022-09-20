import logging
import math
import random
import subprocess

import numpy as np
from deap import creator

from DeepJanus.core.log_setup import get_logger
from DeepJanus.core.misc import evaluate_sparseness
from DeepJanus.core.archive import Archive
from DeepJanus.core.individual import Individual

from torchvision.transforms import ToTensor
from DeepJanus.SEDE_config import SEDEConfig
import itertools
import json
import os
import shutil
import time
from os.path import join, isfile, exists
import cv2
import subprocess as sp
import pathlib as pl
from searchModule import processX, processImage, generateHeatMap, testModelForImg, doDistance, labelImage, \
    getPosePath
from imports import torch, Variable, transforms, entropy, dlib, dirname, basename, Image
import config as cfg
blenderPath = cfg.blenderPath
log = get_logger(__file__)

class SEDEIndividual(Individual):
    counter = 0

    def __init__(self, m1: list, config: SEDEConfig, archive: Archive):
        super().__init__()
        SEDEIndividual.counter += 1
        self.name = f'ind{str(SEDEIndividual.counter)}'
        self.name_ljust = self.name.ljust(6)
        self.config = config
        self.archive = archive
        self.sparseness = None
        self.aggregate = None
        self.params = m1
        self.F1 = None
        self.F2 = None
        self.imgPath = None
        self.heatmap = None
        self.DNNResult = None
        self.faceFound = None
        self.members_distance = None
        self.inCluster = False
        self.seed: SEDEIndividual
        self.generate()

    def evaluate(self):
        if self.needs_evaluation():
            layersHM, self.entropy = generateHeatMap(self.imgPath, self.config.caseFile["DNN"], self.config.caseFile["datasetName"], self.config.caseFile["outputPath"],
                                                False, None, None, self.config.caseFile["imgExt"], None)
            self.heatmap = layersHM[int(self.config.caseFile["selectedLayer"].replace("Layer", ""))]
            self.DNNResult, pred = testModelForImg(self.config.caseFile["DNN"], labelImage(self.imgPath), self.imgPath, self.config.caseFile)
            #self.F2 = (1 - self.entropy) if self.DNNResult else -0.1

            self.members_distance = doDistance(self.config.centroid, self.heatmap, "Euc")/ self.config.caseFile["CR"]
            self.F2 = self.members_distance
            self.sparseness = evaluate_sparseness(self, self.archive, self.config.upper_bound, self.config.lower_bound)
            #self.F1 = self.sparseness - (self.config.K_SD * self.members_distance)
            self.F1 = self.sparseness
            self.inCluster = True if (self.members_distance <= 1) else False
        #log.info(f'evaluated {self}')
        return self.F1, self.F2

    def clone(self) -> 'SEDEIndividual':
        res: SEDEIndividual = creator.Individual(self.params, self.config, self.archive)
        res.seed = self.seed
        #log.info(f'cloned to {res} from {self}')
        return res

    def semantic_distance(self, i2: 'SEDEIndividual'):
        """
        this distance exploits the behavioral information (member.distance_to_boundary)
        so it will compare distances for members on the same boundary side
        :param i2: Individual
        :return: distance
        """
        i1 = self

        i1_posi, i1_nega = i1.members_by_sign()
        i2_posi, i2_nega = i2.members_by_sign()

        return np.mean([i1_posi.distance(i2_posi), i1_nega.distance(i2_nega)])

    def mutate(self):
        res = self
        condition = False
        SEDEIndividual.counter += 1
        self.name = f'ind{str(SEDEIndividual.counter)}'
        self.name_ljust = self.name.ljust(6)
        self.sparseness = None
        self.aggregate = None
        self.F1 = None
        self.F2 = None
        self.imgPath = None
        self.heatmap = None
        self.DNNResult = None
        self.faceFound = None
        self.members_distance = None
        self.inCluster = False
        params = self.params
        while not condition:
            self.params = []
            for i, xl, xu in zip(range(len(params)), self.config.lower_bound, self.config.upper_bound):
                if random.random() < 0.5:
                    self.params.append(random.uniform(xl, xu))
                else:
                    self.params.append(params[i])
            #res: SEDEIndividual = creator.Individual(params, self.config, self.archive)
            #res.seed = self
            self.generate()
            if self.faceFound is not None and self.faceFound:
                condition = True
            #    log.info("Mutated")
            else:
                log.info("Mutation failed .. re-attempting")
        return self

    def needs_evaluation(self):
        return self.DNNResult is None

    def clear_evaluation(self):
        self.DNNResult = None
        self.F1 = None
        self.F2 = None


    def is_valid(self):
        return self.faceFound

    def to_tuple(self):
        import numpy as np
        barycenter = np.mean(self.control_nodes, axis=0)[:2]
        return barycenter
    def export(self, outPath):
        if not exists(outPath):
            os.makedirs(outPath)
        if isfile(join(outPath, processX(self.params) + ".png")):
        #    shutil.copy(self.imgPath, join(outPath, processX(self.params) + "_" + str(self.config.duplicateCount) + ".png"))
        #    self.config.duplicateCount += 1
            return False
        else:
            shutil.copy(self.imgPath, join(outPath, processX(self.params) + ".png"))
            return True
    def generate(self):
        t1 = time.time()
        if self.config.caseFile["caseStudy"] == "HPDH":
            self.generateHuman()
        elif self.config.caseFile["caseStudy"] == "HPDF":
            self.generateIndividual()
        else:
            self.generateFLDIndividual()

        t2 = time.time()
        print("Image Generation: ", str(t2-t1)[0:5], end="\r")
    def generateIndividual(self):
        x = self.params
        SimDataPath = self.config.caseFile["SimDataPath"]
        outPath = self.config.caseFile["outputPath"]
        model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
        m = random.randint(0, len(pose_file) - 1)
        m = int(math.floor(x[len(x) - 1]))
        imgPath = join(SimDataPath, processX(x))
        filePath = os.path.join(pl.Path(__file__).parent.parent.resolve(), "IEE_V1", "ieekeypoints.py")
        # filePath = "./ieekeypoints.py"
        t1 = time.time()
        if not isfile(imgPath + ".png"):
            # ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
            # print(blenderPath)
            ls = subprocess.run(
                [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
                 "--path",
                 str(join(pl.Path(__file__).parent.parent.resolve(), "IEEPackage")), "--model_folder",
                 str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder),
                 "--pose_file",
                 str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]),
                 "--imgPath",
                 # str(imgPath)])
                 str(imgPath)], stdout=subprocess.PIPE)
            # str(imgPath)], capture_output=True, text=True, shell=True).stderr.strip("\n")
            #log.info(ls)
            #log.info(ls.stdout)
            # print(process.stderr)
        # process.wait()
        t2 = time.time()
        # print("Image Generation: ", str(t2-t1)[0:5], end="\r")
        # print(imgPath)
        img = cv2.imread(imgPath + ".png")
        if img is None:
            log.debug("image not found, not processed")
            self.faceFound = None
        if len(img) > 128:
            self.faceFound = processImage(imgPath + ".png",
                                          join(pl.Path(__file__).parent.parent.resolve(), "IEEPackage", "clsdata",
                                               "mmod_human_face_detector.dat"))
        else:
            self.faceFound = True
        # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
        # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
        #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
        #                                                   label_file[m])
        # print("Image processing", str(time.time()-t2)[0:5])
        #return imgPath, faceFound
        self.imgPath = join(SimDataPath, processX(x)) + ".png"

    def generateHuman(self):
        x = self.params
        caseFile = self.config.caseFile
        # x = [head0/1/2, lamp_col0/1/2, lamp_loc0/1/2, lamp_dir0/1/2, cam2, age, eye_hue, eye_iris, eye_sat,
        # eye_val, skin_freckle, skin_oil, skin_veins, eye_color, gender]
        # x[0:12] -- Scenario/Person
        # x[13:23] -- MBLab/Human
        #x[19] = math.floor(x[19])
        #print(x)
        SimDataPath = caseFile["SimDataPath"]
        imgPath = join(SimDataPath, str(self.config.globalCounter), processX(x))
        t1 = time.time()
        if not isfile(join(SimDataPath, processX(x)) + ".png"):
            # output = os.path.join(pl.Path(__file__).resolve().parent.parent.parent, 'snt_simulator', 'data', 'mblab')
            script = os.path.join(pl.Path(__file__).parent.parent.resolve(),
                                  "IEE_V2", "mblab-interface", "scripts", "snt_face_dataset_generation.py")
            #print("calling script", script)
            data_path = join(pl.Path(__file__).parent.parent.resolve(), "IEE_V2", "mblab_asset_data")
            #print(blenderPath)
            cmd = [str(blenderPath), "-b", "--log-level", str(0), "-noaudio", "--python", script, "--", str(data_path), "-l", "debug", "-o",
                   f"{imgPath}", "--render", "--studio-lights"]  # generate MBLab character
            try:
                devnull = open(join(SimDataPath, str(self.config.globalCounter) + "_MBLab_log.txt"), 'w')
                #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
                sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)

            except Exception as e:
                log.error("Error in MBLab creating")
                log.error(e)
                #exit(0)
            filePath = os.path.join(pl.Path(__file__).parent.parent.resolve(), "IEE_V2", "ieekeypoints2.py")
            try:
                devnull = open(join(SimDataPath, str(self.config.globalCounter) + "_Blender_log.txt"), 'w')
                #print(filePath)
                #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
                cmd = [str(blenderPath), "--background", "-noaudio", "--verbose", str(0), "--python", str(filePath), "--",
                     "--imgPath", str(imgPath)]
                #sp.call(cmd, env=os.environ, shell=True, stdout=devnull, stderr=devnull)
                sp.call(cmd, env=os.environ, stdout=devnull, stderr=devnull)
                # str(imgPath)], stdout=subprocess.PIPE)
                shutil.copy(imgPath + ".png", join(SimDataPath, processX(x) + ".png"))
                shutil.copy(imgPath + ".npy", join(SimDataPath, processX(x) + ".npy"))
                shutil.rmtree(join(SimDataPath, str(self.config.globalCounter)))
            except Exception as e:
                log.error("error in Blender scenario creation")
                log.error(e)
                #exit(0)
            # print(ls)
            self.config.globalCounter += 1
        imgPath = join(SimDataPath, processX(x) + ".png")
        img = cv2.imread(imgPath)
        t2 = time.time()
        #print("Image Generation: ", str(t2-t1)[0:5], end="\r")
        if img is None:
            log.debug("image not found, not processed")
            #exit(0)
            self.faceFound = None
        else:
            if len(img) > 128:
                self.faceFound = processImage(imgPath,
                                          join(pl.Path(__file__).parent.parent.resolve(), "IEEPackage", "clsdata",
                                               "mmod_human_face_detector.dat"))
            else:
                self.faceFound = True
        #shutil.copy(join(SimDataPath, processX(x)) + ".png", join(caseFile["GI"], str(caseFile["ID"]), "images", processX(x) + ".png"))
        self.imgPath = imgPath

    def generateFLDIndividual(self):
        counter = self.config.globalCounter
        x = self.params
        SimDataPath = self.config.caseFile["SimDataPath"]
        outPath = self.config.caseFile["outputPath"]
        path = join(pl.Path(__file__).parent.parent.resolve(), "IEEPackage")
        model_folder, label_folder, pose_folder, model_file, label_file, pose_file = getPosePath()
        m = random.randint(0, len(pose_file) - 1)
        m = int(math.floor(x[len(x) - 1]))
        DNNResult = None
        #print(pose_file[m])
        #print(label_file[m])
        #print(model_file[m])
        imgPath = join(SimDataPath, processX(x))
        filePath = os.path.join(pl.Path(__file__).parent.parent.resolve(), "IEE_V1", "ieekeypoints.py")
        # filePath = "./ieekeypoints.py"
        t1 = time.time()
        if not isfile(imgPath + ".png"):
            # ls = subprocess.run(['ls', '-a'], capture_output=True, text=True).stdout.strip("\n")
            ls = subprocess.run(
                [str(blenderPath), "--background", "--verbose", str(0), "--python", str(filePath), "--", "--path",
                 str(path), "--model_folder",
                 str(model_folder), "--label_folder", str(label_folder), "--pose_folder", str(pose_folder),
                 "--pose_file",
                 str(pose_file[m]), "--label_file", str(label_file[m]), "--model_file", str(model_file[m]), "--imgPath",
                 # str(imgPath)], stdout=subprocess.PIPE)
                 str(imgPath)], capture_output=True, text=True).stderr.strip("\n")
            # print(ls)
            # print(process.stdout)
            # print(process.stderr)
        # process.wait()
        t2 = time.time()
        # print("Image Generation", str(t2-t1)[0:5])
        # print(imgPath)
        img = cv2.imread(imgPath + ".png")
        new_img = None
        new_label = None
        if img is None:
            print("image not found, not processed")
            self.faceFound = None
            self.DNNResult = None
            new_img = None
            return
        else:
            if len(img) > 128:
                self.faceFound, new_img, new_label = self.processImageFLD(imgPath + ".png", join(pl.Path(__file__).parent.parent.resolve(),
                                                                                    "IEEPackage", "clsdata", "mmod_human_face_detector.dat"))

                configData = {"data": new_img, "label": new_label}
                torch.save(configData, imgPath + ".pt")
                if not self.faceFound:
                    return
            else:
                configData = torch.load(imgPath + ".pt")
                new_img = configData["data"]
                new_label = configData["label"]
                self.faceFound = True
        imgPath = imgPath + ".png"
        if new_img is not None:
            # if not isfile(imgPath + ".pt"):
            # else:
            def update(img, x_p, y_p, x_t=0, y_t=0):
                height, width = img.shape[0], img.shape[1]
                for idx in [-1, 0, 1]:
                    px = max(min(x_p + idx, width - 1), 0)
                    if x_t > 0 and y_t > 0:
                        tx = max(min(x_t + idx, width - 1), 0)
                    for jdx in [-1, 0, 1]:
                        py = max(min(y_p + jdx, height - 1), 0)
                        if x_t > 0 and y_t > 0:
                            ty = max(min(y_t + jdx, height - 1), 0)

                        if width > py > 0 and height > px > 0:
                            img[py, px, 0] = 255
                            img[py, px, 1] = 155
                            img[py, px, 2] = 0

                        if x_t > 0 and y_t > 0:

                            if width > ty > 0 and height > tx > 0:
                                img[ty, tx, 0] = 0
                                img[ty, tx, 1] = 0
                                img[ty, tx, 2] = 255
                        # else:
                        #    print("unlabelled image")
                return img
            def forward_hook(self, input, output):
                # print("forward hook..")
                self.X = input[0]
                self.Y = output
            def ieeRegister(model):
                model.conv2d_1.register_forward_hook(forward_hook)
                model.conv2d_2.register_forward_hook(forward_hook)
                model.maxpool_1.register_forward_hook(forward_hook)
                model.conv2d_3.register_forward_hook(forward_hook)
                model.conv2d_4.register_forward_hook(forward_hook)
                model.maxpool_2.register_forward_hook(forward_hook)
                model.conv2d_5.register_forward_hook(forward_hook)
                model.conv2d_6.register_forward_hook(forward_hook)
                model.conv2d_trans_1.register_forward_hook(forward_hook)
                model.conv2d_trans_2.register_forward_hook(forward_hook)
                return model
            def get_ave_xy(hmi, n_points=64, thresh=0.0):
                assert n_points > 1
                ind = hmi.argsort(axis=None)[-n_points:]  ## pick the largest n_points
                topind = np.unravel_index(ind, hmi.shape)
                index = np.unravel_index(hmi.argmax(), hmi.shape)
                i0, i1, hsum = 0, 0, 0
                for ind in zip(topind[0], topind[1]):
                    h = hmi[ind[0], ind[1]]
                    hsum += h
                    i0 += ind[0] * h
                    i1 += ind[1] * h
                i0 /= hsum
                i1 /= hsum
                if hsum / n_points <= thresh:
                    i0, i1 = -1, -1
                return [i1, i0]
            def transfer_xy_coord(hm, n_points=64, thresh=0.2):
                assert len(hm.shape) == 3
                Nlandmark = hm.shape[0]
                # est_xy = -1*np.ones(shape transfer= (Nlandmark, 2))
                est_xy = []
                for i in range(Nlandmark):
                    hmi = hm[i, :, :]
                    est_xy.append(get_ave_xy(hmi, n_points, thresh))
                return est_xy  ## (Nlandmark, 2)
            def transfer_target(y_pred, thresh=0, n_points=64):
                y_pred_xy = []
                for i in range(y_pred.shape[0]):
                    hm = y_pred[i]
                    y_pred_xy.append(transfer_xy_coord(hm, n_points, thresh))
                return (np.array(y_pred_xy))
            def ieeBackKP(predict_cpu, KPindex):
                for i in range(0, len(predict_cpu)):
                    for j in range(0, len(predict_cpu[i])):
                        if j != KPindex:
                            predict_cpu[i][j] = 0
                return predict_cpu
            def returnHeatmap(model, Alex, HM):
                if not Alex:
                    heatmaps = [0] * 10
                    if HM:
                        heatmaps[0] = model.conv2d_1.HM.detach()
                        heatmaps[1] = model.conv2d_2.HM.detach()
                        heatmaps[2] = model.maxpool_1.HM.detach()
                        heatmaps[3] = model.conv2d_3.HM.detach()
                        heatmaps[4] = model.conv2d_4.HM.detach()
                        heatmaps[5] = model.maxpool_2.HM.detach()
                        heatmaps[6] = model.conv2d_5.HM.detach()
                        heatmaps[7] = model.conv2d_6.HM.detach()
                        heatmaps[8] = model.conv2d_trans_1.HM.detach()
                        heatmaps[9] = model.conv2d_trans_2.HM.detach()
                    else:
                        heatmaps[0] = model.conv2d_1.Y.detach()
                        heatmaps[1] = model.conv2d_2.Y.detach()
                        heatmaps[2] = model.maxpool_1.Y.detach()
                        heatmaps[3] = model.conv2d_3.Y.detach()
                        heatmaps[4] = model.conv2d_4.Y.detach()
                        heatmaps[5] = model.maxpool_2.Y.detach()
                        heatmaps[6] = model.conv2d_5.Y.detach()
                        heatmaps[7] = model.conv2d_6.Y.detach()
                        heatmaps[8] = model.conv2d_trans_1.Y.detach()
                        heatmaps[9] = model.conv2d_trans_2.Y.detach()
                else:
                    k = 0
                    sizee = len(model.features) + len(model.classifier)
                    heatmaps = [0] * sizee
                    for i in range(0, len(model.features)):
                        if HM:
                            heatmaps[k] = model.features[i].HM.detach()
                        else:
                            heatmaps[k] = model.features[i].Y.detach()
                        k += 1
                    for i in range(0, len(model.classifier)):
                        if HM:
                            heatmaps[k] = model.classifier[i].HM.detach()
                        else:
                            heatmaps[k] = model.classifier[i].Y.detach()
                        k += 1
                return heatmaps
            data_transform = transforms.Compose([ToTensor()])
            inputs = data_transform(new_img).unsqueeze(0)
            labels = new_label
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            model = self.config.caseFile["DNN"]
            model = ieeRegister(model)
            predict = model(inputs)
            predict_cpu = predict.cpu()
            predict_cpu = predict_cpu.detach().numpy()
            predict_xy = transfer_target(predict_cpu)
            inputs_cpu = inputs.cpu()
            inputs_cpu = inputs_cpu.detach().numpy()
            num_sample = 1
            diff = np.square(new_label - predict_xy)
            sum_diff = np.sqrt(diff[:, :, 0] + diff[:, :, 1])
            wlabel = []
            avg_error = np.sum(sum_diff[0]) / len(sum_diff[0])
            worst_KP = 0
            label = 0
            worst_label = 0
            kps = []
            for KP in sum_diff[0]:
                kps.append(label)
                if KP > worst_KP:
                    worst_KP = KP
                    worst_label = label
                label += 1
            wlabel.append(worst_label)
            kps = wlabel
            # print(kps)
            # print(wlabel)
            i = 0
            for idx in range(num_sample):
                img = inputs_cpu[idx] * 255.
                img = img[0, :]
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                xy = predict_xy[idx]
                lab_xy = labels
                # print(lab_xy)
                for kp in kps:
                    # print(kp)
                    x_p = int(xy[kp, 0] + 0.5)
                    y_p = int(xy[kp, 1] + 0.5)
                    x_t = int(lab_xy[kp, 0] + 0.5)
                    y_t = int(lab_xy[kp, 1] + 0.5)
                    if x_t == 0 and y_t == 0:
                        self.faceFound = False
                        new_img = None
                        new_label = None
                    img = update(img, x_p, y_p, x_t, y_t)
                cv2.imwrite(imgPath, img)
            npyFile = np.load(imgPath.split(".png")[0] + ".npy", allow_pickle=True)
            # inputs =new_img[0]
            cp_labels = npyFile
            labels_gt = cp_labels.item()["label"]
            if new_img is None:
                self.faceFound = None
                DNNResult = None
            predict_cpu = ieeBackKP(predict_cpu, worst_label)
            # predict_cpu = HeatmapModule.ieeBackParts(predict_cpu, area)
            tAF = torch.from_numpy(predict_cpu[0]).type(torch.FloatTensor)
            if torch.cuda.is_available():
                tAF = Variable(tAF).cuda()
            else:
                tAF = Variable(tAF).cpu()
            model.relprop(tAF)
            layersHM = returnHeatmap(model, False, True)
            # print(sum_diff[0])
            en = np.exp(sum_diff[0]) / np.sum(np.exp(sum_diff[0]))
            self.entropy = entropy(en, base=27)  ##fixme base=num_classes
            self.heatmap = layersHM[int(self.config.caseFile["selectedLayer"].replace("Layer", ""))]
            if avg_error > 4:
                DNNResult = False
            else:
                DNNResult = True
            self.imgPath = imgPath
            self.DNNResult = DNNResult
            #self.faceFound = faceFound
            #self.F2 = (1 - self.entropy) if self.DNNResult else -0.1

            self.members_distance = doDistance(self.config.centroid, self.heatmap, "Euc")/ self.config.caseFile["CR"]
            self.F2 = self.members_distance
            self.sparseness = evaluate_sparseness(self, self.archive, self.config.upper_bound, self.config.lower_bound)
            #self.F1 = self.sparseness - (self.config.K_SD * self.members_distance)
            self.F1 = self.sparseness
            #if self.members_distance/self.config.caseFile["CR"] <= 1:
            if self.members_distance <= 1:
                self.inCluster = True

        # generator = ieeKP.IEEKPgenerator(model_folder, pose_folder, label_folder)
        # imgPath = generator.generate_with_single_processor(width, height, head, lamp_dir, lamp_col, lamp_loc, lamp_eng,
        #                                                   cam_loc, cam_dir, SimDataPath, pose_file[m], model_file[m],
        #                                                   label_file[m])
        # print("Image processing", str(time.time()-t2)[0:5])

        #shutil.copy(imgPath.split(".png")[0] + ".pt",
        #            os.path.join(os.path.dirname(imgPath), "S" + str(counter - 1) + ".pt"))

    def processImageFLD(self, imgPath, dlibPath):
        #global counter
        img = cv2.imread(imgPath)
        npPath = join(dirname(imgPath), basename(imgPath).split(".png")[0] + ".npy")
        configFile = np.load(npPath, allow_pickle=True)
        labelFile = configFile.item()['label']

        face_detector = dlib.cnn_face_detection_model_v1(dlibPath)
        faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1)
        # img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return
        if len(faces) < 1:
            #self.faceFound = False
            #return
            return False, None, None
        big_face = -np.inf
        mx, my, mw, mh = 0, 0, 0, 0
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            if w * h > big_face:
                big_face = w * h
                mx, my, mw, mh = x, y, w, h
        sw_0 = max(mx - 25 // 2, 0)
        sw_1 = min(mx + mw + 25 // 2, new_img.shape[1])  # empirical
        sh_0 = max(my - 25 // 2, 0)
        sh_1 = min(my + mh + 25 // 2, new_img.shape[0])  # empirical
        assert sh_1 > sh_0
        assert sw_1 > sw_0
        label_arr = []
        iee_labels = [18, 22, 23, 27, 28, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 69, 43, 44, 45, 46, 47, 48, 70, 49,
                      52,
                      55, 58]
        for ky in iee_labels:
            if ky in labelFile:
                coord = [labelFile[ky][0], new_img.shape[0] - labelFile[ky][1]]
                label_arr.append(coord)  # (-1,-1) means the keypoint is invisible
            else:
                label_arr.append([0, 0])  # label does not exist
        new_label = np.zeros_like(np.array(label_arr))
        new_label[:, 0] = np.array(label_arr)[:, 0] - sw_0
        new_label[:, 1] = np.array(label_arr)[:, 1] - sh_0
        new_label[new_label < 0] = 0
        new_label[np.array(label_arr)[:, 0] == -1, 0] = -1
        new_label[np.array(label_arr)[:, 1] == -1, 1] = -1
        big_face = new_img[sh_0:sh_1, sw_0:sw_1]
        width_resc = float(128) / big_face.shape[0]
        height_resc = float(128) / big_face.shape[1]
        new_label2 = np.zeros_like(new_label)
        new_label2[:, 0] = new_label[:, 0] * width_resc
        new_label2[:, 1] = new_label[:, 1] * height_resc
        labelFile = new_label2

        new_img = cv2.resize(big_face, (128, 128), interpolation=cv2.INTER_CUBIC)
        # print(new_img.shape)
        x_data = new_img
        x_data = np.repeat(x_data[:, :, np.newaxis], 3, axis=2)
        img = x_data
        cv2.imwrite(imgPath, img)
        #label = labelImage(imgPath)
        #label_dir = os.path.join(os.path.dirname(os.path.dirname(imgPath)), label)
        #if not os.path.exists(label_dir):
        #    os.makedirs(label_dir)
        #newImgPath = os.path.join(label_dir, "S" + str(counter) + ".png")
        #shutil.copy(imgPath.split(".png")[0] + ".npy",
        #            os.path.join(os.path.dirname(imgPath), "S" + str(counter) + ".npy"))
        #cv2.imwrite(newImgPath, img)
        #counter += 1
        return True, new_img, new_label2
    