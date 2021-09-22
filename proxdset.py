import pickle
import numpy as np
import datetime as datetime
import skimage.io as io
import os
import math
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import glob
import torch.optim as optim
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import hashlib
import pickle
import plyfile
import os
import os.path as osp
import torch
import cv2
import numpy as np
import json
from plyfile import PlyData


def read_skeleton(skeleton_fn):
    # skeleton_fn: the path of skeleton points data
    skeletons = []
    with open(skeleton_fn) as skeleton_file:
        #prox dataset save skeletion as json file
        data = json.load(skeleton_file)
    # data includes many dictionaries, including body, face, hand, leg, we just need body skeleton
    Bodies = data['Bodies']
    count = 0
    for Dict in Bodies:
        Joints = Dict['Joints']
        Joint_keys = Joints.keys()
        #traverse the list
        for joint_key in Joint_keys:
            joint = Joints[joint_key]
            position = joint['Position']
            skeletons.append(position)
    #convert list into array        
    skeletons = np.array(skeletons)
    return skeletons


class D2_D3_PROX(Dataset):
    """Dataloder for the PROX datasets"""
    #record_sections -> the path of record sections
    #keyp_sections -> keyp sections path
    #tag -> personal tag 
    #seq_in -> number of input time stpes 
    #seq_out -> number of time steps to be predicted 
    #load_img -> load images? 
    #load_depth -> load depth data? 
    #img_resize -> scale down images, saves memoery also the deafult keeps aspect rartio
    #save -> save for faster preload in the future
    def __init__(
            self, record_sections, keyp_sections, tag='', seq_in=5, seq_out=10, load_img=True, load_depth=False,
            img_resize=(90, 160),
            save=True):
        super(D2_D3_PROX, self).__init__()
        self.load_img = load_img
        self.load_depth = load_depth
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.save = save
        self.img_resize = img_resize
        self.tag = tag
        self.record_sections = record_sections
        self.keyp_sections = keyp_sections
        # This is generating a random filename for checkpoint of trained model so we don't need to retrain with the same configuration
        _hash = '-'.join(record_sections) + str(tag) + str(seq_in) + str(seq_out) + str(load_img) + str(
            load_depth) + str(img_resize[0]) + str(img_resize[1])
        savefile = './' + hashlib.sha256(_hash.encode()).hexdigest() + '_PROX.pkl'
        print("Save file is:", savefile)

        if save:
            #if save = True, we can just import the dataset saved as pkl file.
            if os.path.exists(savefile):
                print("Save file:", savefile, " exists... loading it")
                with open(savefile, 'rb') as f:
                    data = pickle.load(f)
                self._x = data['_x']
                self._xA = data['_xA']
                self._y = data['_y']
                if load_img:
                    self._vrgb = data['_vrgb']
                if load_depth:
                    self._vdepth = data['_vdepth']
            else:
                # This need to be computed once, the Adjacency matrix of the skeleton
                LIMBS = [
                    (0, 1),  # SpineBase -> SpineMid
                    (1, 20),  # SpineMid -> SpineShoulder
                    (20, 2),  # SpineShoulder -> neck
                    (2, 3),  # neck -> head
                    (20, 4),  # SpineShoulder -> left_shoulder
                    (4, 5),  # left_shoulder -> left_elbow
                    (5, 6),  # left_elbow -> left_wrist
                    (6, 7),  # left_wrist -> left_hand
                    (0, 12),  # SpineBase -> left_hip
                    (12, 13),  # left_hip -> left_knee
                    (13, 14),  # left_knee -> left_ankle
                    (14, 15),  # left_ankle -> left_foot
                    (20, 8),  # SpineShoulder -> right_shoulder
                    (8, 9),  # right_shoulder -> right_elbow
                    (9, 10),  # right_elbow -> right_wrist
                    (10, 11),  # right_wrist -> right_hand
                    (0, 16),  # SpineBase -> right_hip
                    (16, 17),  # right_hip -> right_knee
                    (17, 18),  # right_knee -> right_ankle
                    (18, 19),  # right_ankle -> right_foot
                ]
                A = np.zeros((21, 21))
                for i, j in LIMBS:
                    A[i, j] = 1
                    A[j, i] = 1
                #generate G matrix according to the LIMBS
                G = nx.from_numpy_matrix(A)
                Anorm = nx.normalized_laplacian_matrix(G).toarray()  # It's the same adj matrix for all

                _x = []
                _xA = []
                if load_img:
                    _vrgb = []
                if load_depth:
                    _vdepth = []
                _y = []
                
                for keyp_section in keyp_sections: #Loop through keysections
                    #traverse the list
                    _d2 = []
                    frames_list = os.listdir(keyp_section)
                    #sort the filenames for synchronization
                    frames_list.sort()
                    for frame in frames_list:
                        #read each frame of video
                        keyp_path = osp.join(keyp_section, frame)
                        d2 = np.load(keyp_path).T
                        _d2.append(d2[None, ...])

                    kk = 0
                    for k in range(0, len(frames_list) - (seq_in + seq_out), 10):
                        kk = k
                    #processing bar
                    pbar = tqdm(total=kk)

                    for i in range(0, len(frames_list) - (seq_in + seq_out), 10):
                        _x.append(torch.from_numpy(np.concatenate(_d2[i:i + seq_in], axis=0)).type(torch.float32))
                        pbar.update(1)
                    pbar.close()

                for record_section in record_sections: #Loop through record sections
                    if load_img:
                        _rgb = []
                    if load_depth:
                        _depth = []
                    _d2A = []
                    _d3 = []
                    frames_list = os.listdir(record_section + 'Skeleton')
                    frames_list.sort()
                    for frame in frames_list: #Load frames first
                        # read 3d skeleton
                        skeleton_path = osp.join(record_section + 'Skeleton', frame)
                        if os.path.isdir(skeleton_path):
                            continue
                        d3 = read_skeleton(skeleton_path)
                        if d3.ndim < 2:
                            #for filtering the skeleton data with 0 point in PROX dataset
                            continue
                        d3 = d3[:21]
                        _d3.append(d3[None, ...])
                        _d2A.append(A[None, ...])

                        if load_img:
                            image_path = osp.join(record_section + 'Color', frame[:-4] + 'jpg')
                            rgb = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
                                             img_resize, interpolation=cv2.INTER_LANCZOS4) / 255.0
                        if load_depth:
                            depth_path = osp.join(record_section + 'Depth', frame[:-4] + 'png')
                            depth = cv2.resize(cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB),
                                               img_resize, interpolation=cv2.INTER_LANCZOS4) / 255.0

                        if load_img:
                            _rgb.append(rgb[None, ...].transpose(0, 3, 1, 2))

                        if load_depth:
                            _depth.append(depth[None, ...].transpose(0, 3, 1, 2))

                    kk = 0
                    for k in range(0, len(_d3) - (seq_in + seq_out), 10):
                        kk = k

                    pbar = tqdm(total=kk)

                    #Create the input/output sequences
                    for i in range(0, len(_d3) - (seq_in + seq_out), 10):
                        if load_img:
                            _vrgb.append(
                                torch.from_numpy(np.concatenate(_rgb[i:i + seq_in], axis=0)).type(torch.float32))
                        if load_depth:
                            _vdepth.append(
                                torch.from_numpy(np.concatenate(_depth[i:i + seq_in], axis=0)).type(torch.float32))

                        _xA.append(torch.from_numpy(np.concatenate(_d2A[i:i + seq_in], axis=0)).type(torch.float32))
                        _y.append(torch.from_numpy(np.concatenate(_d3[i + seq_in:i + seq_in + seq_out], axis=0)).type(
                            torch.float32))
                        pbar.update(1)
                    pbar.close()

                print('Loading done.')
                print(len(_x))
                print(len(_y))
                print(len(_xA))
                self._x = _x
                self._xA = _xA
                self._y = _y
                if load_img:
                    self._vrgb = _vrgb
                if load_depth:
                    self._vdepth = _vdepth

                if save:
                    with open(savefile, 'wb') as f:
                        data = {}
                        data['_x'] = _x
                        data['_xA'] = _xA
                        data['_y'] = _y
                        if load_img:
                            data['_vrgb'] = _vrgb
                        if load_depth:
                            data['_vdepth'] = _vdepth
                        pickle.dump(data, f)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):

        output = [self._x[index], self._xA[index], self._y[index]]
        if self.load_img:
            output.append(self._vrgb[index])
        if self.load_depth:
            output.append(self._vdepth[index])
        return output