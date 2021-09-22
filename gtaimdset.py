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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import hashlib




class D2_D3_GTA_IM(Dataset):
    """Dataloder for the GTA IM datasets"""
    #dpaths -> the path of data
    #tag -> personal tag 
    #seq_in -> number of input time stpes 
    #seq_out -> number of time steps to be predicted 
    #load_img -> load images? 
    #load_depth -> load depth data? 
    #img_resize -> scale down images, saves memoery also the deafult keeps aspect rartio
    #save -> save for faster preload in the future
    def __init__(
        self, dpaths, tag='',seq_in=5, seq_out=10,load_img = True, load_depth =False, img_resize=(90,160),
        save= True):
        super(D2_D3_GTA_IM, self).__init__()
        
        self.load_img = load_img
        self.load_depth = load_depth
        self.seq_in = seq_in
        self.seq_out= seq_out
        self.save = save
        self.img_resize = img_resize
        self.tag = tag
        
        #This is for faster loading of data, if processed before we just load it 
        _hash = '-'.join(dpaths)+str(tag)+str(seq_in)+str(seq_out)+str(load_img)+str(load_depth)+str(img_resize[0])+str(img_resize[1])
        savefile = './'+hashlib.sha256(_hash.encode()).hexdigest()+'.pkl'
        print("Save file is:", savefile)
        
        if save:
            if os.path.exists(savefile):
                print("Save file:",savefile," exists... loading it")
                with open(savefile,'rb') as f: 
                    data = pickle.load(f)
                self._x = data['_x']
                self._xA = data['_xA']
                self._y = data['_y']
                if load_img:
                    self._vrgb = data['_vrgb']
                if load_depth:
                    self._vdepth = data['_vdepth']
                    
            else:

                #This need to be computed once, the Adjacency matrix of the skeleton 
                LIMBS = [
                    (0, 1),  # head_center -> neck
                    (1, 2),  # neck -> right_clavicle
                    (2, 3),  # right_clavicle -> right_shoulder
                    (3, 4),  # right_shoulder -> right_elbow
                    (4, 5),  # right_elbow -> right_wrist
                    (1, 6),  # neck -> left_clavicle
                    (6, 7),  # left_clavicle -> left_shoulder
                    (7, 8),  # left_shoulder -> left_elbow
                    (8, 9),  # left_elbow -> left_wrist
                    (1, 10),  # neck -> spine0
                    (10, 11),  # spine0 -> spine1
                    (11, 12),  # spine1 -> spine2
                    (12, 13),  # spine2 -> spine3
                    (13, 14),  # spine3 -> spine4
                    (14, 15),  # spine4 -> right_hip
                    (15, 16),  # right_hip -> right_knee
                    (16, 17),  # right_knee -> right_ankle
                    (14, 18),  # spine4 -> left_hip
                    (18, 19),  # left_hip -> left_knee
                    (19, 20)  # left_knee -> left_ankle
                ]

                A = np.zeros((21,21))
                for i,j in LIMBS: 
                    A[i,j] = 1
                    A[j,i] = 1

                G = nx.from_numpy_matrix(A)
                Anorm = nx.normalized_laplacian_matrix(G).toarray() #It's the same adj matrix for all 


                _x = []
                _xA = []
                if load_img:
                    _vrgb = []
                if load_depth:
                    _vdepth = []
                _y = [] 

                for dpath in dpaths: #Loop over all data paths

                    info = pickle.load(open(dpath + 'info_frames.pickle', 'rb')) #Load the compressed data 
                    info_npz = np.load(dpath+'info_frames.npz')


                    if load_img:
                        _rgb = []
                    if load_depth:
                        _depth = []
                    _d2 = [] 
                    _d2A = []
                    _d3 = []
                    for fm_id   in range(len(info)): #Load per frame info 

                        if load_img:
                            rgb = cv2.resize(cv2.cvtColor(cv2.imread(dpath+'{:05d}'.format(fm_id)+'.jpg'), cv2.COLOR_BGR2RGB),
                                         img_resize, interpolation = cv2.INTER_LANCZOS4)/255.0


                        if load_depth:
                            depth = cv2.resize(cv2.cvtColor(cv2.imread(dpath+'{:05d}'.format(fm_id)+'.png'), cv2.COLOR_BGR2RGB),
                                         img_resize, interpolation = cv2.INTER_LANCZOS4)/255.0
                        #Skip images for now as we don't have all of them
                        d2 = info_npz['joints_2d'][fm_id] 
                        d3 = info_npz['joints_3d_cam'][fm_id]

                        if load_img:
                            _rgb.append(rgb[None,...].transpose(0,3,1,2))
                        if load_depth:
                            _depth.append(depth[None,...].transpose(0,3,1,2))
                        _d2.append(d2[None,...])
                        _d2A.append(A[None,...])
                        _d3.append(d3[None,...])


                    #Create the sequences, using a moving window of (seq_in_seq_out)
                    kk =0 
                    for k in range(0,len(info)-(seq_in+seq_out),1):
                        kk =k

                    pbar = tqdm(total=kk) 

                    for i in range(0,len(info)-(seq_in+seq_out),1):
                        _x.append(torch.from_numpy(np.concatenate(_d2[i:i+seq_in],axis=0)).type(torch.float32))
                        if load_img:
                            _vrgb.append(torch.from_numpy(np.concatenate(_rgb[i:i+seq_in],axis=0)).type(torch.float32))
                        if load_depth:
                            _vdepth.append(torch.from_numpy(np.concatenate(_depth[i:i+seq_in],axis=0)).type(torch.float32))

                        _xA.append(torch.from_numpy(np.concatenate(_d2A[i:i+seq_in],axis=0)).type(torch.float32))
                        _y.append(torch.from_numpy(np.concatenate(_d3[i+seq_in:i+seq_in+seq_out],axis=0)).type(torch.float32))
                        pbar.update(1)
                    pbar.close()

                self._x = _x
                self._xA = _xA
                self._y = _y
                if load_img:
                    self._vrgb = _vrgb
                if load_depth:
                    self._vdepth = _vdepth
                    
                if save: #Dump as pickles
                    with open(savefile,'wb') as f : 
                        data = {}
                        data['_x'] = _x 
                        data['_xA'] = _xA 
                        data['_y'] = _y 
                        if load_img:
                            data['_vrgb'] = _vrgb
                        if load_depth:
                            data['_vdepth'] = _vdepth
                        pickle.dump(data,f)
                            


    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):

        output = [self._x[index],self._xA[index],self._y[index]]
        if self.load_img:
            output.append(self._vrgb[index])
        if self.load_depth:
            output.append(self._vdepth[index])
        return output