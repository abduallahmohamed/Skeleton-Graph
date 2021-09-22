import torch
import os
import pickle
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributions as torchdist
import torchvision
from torch.utils.data import dataloader,dataset
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from utils import *
from dataset_gtaim import D2_D3_GTA_IM
from proxdset import D2_D3_PROX
from model import SkeletonGraph
import glob
import hashlib
import open3d as o3d
import cv2
import sys

parser = argparse.ArgumentParser()

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
    (19, 20),  # left_knee -> left_ankle
]

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=3)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--learn_A', action="store_false", default=True,help='Self learning Adjacecny matrix')
parser.add_argument('--video_back', action="store_true", default=False,help='Use Sequence of image embedding')
parser.add_argument('--background_back',action="store_true", default=False, help='Use single image embedding ')

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=5)  #How many corresponds the paper?
parser.add_argument('--pred_seq_len', type=int, default=10)
parser.add_argument('--dataset', default='GTA_IM',
                    help='PROX,GTA_IM')    
parser.add_argument('--im_w', type=int, default=90)
parser.add_argument('--im_h', type=int, default=160)

#loss function sepecific parameters
parser.add_argument('--use_cons', action="store_true", default=False,help='Use consistently loss')
parser.add_argument('--l_norm', type=float, default=0.01,
                    help='weight of norm')
parser.add_argument('--l_cos', type=float, default=0.01,
                    help='weight of cos similarity')

#Training specifc parameters
parser.add_argument('--log_frq', type=int, default=32,
                    help='frequency of logging')
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=600,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=200,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='NA',
                    help='Personal tag epx')
#misc 
parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')
parser.add_argument('--torso_joint', type=int, default=13,
                    help='center of torso, 13 for GTA-IM')

                    
args = parser.parse_args()

#create a unique tag per exp
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only' or k =='torso_joint':
        continue
    args_hash += str(k)+str(v)
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()
args.tag+=args_hash

#dataset settings
datasections = glob.glob('./GTAIMFPS5/*/')
load_img = False
if args.video_back or args.background_back:
    load_img = True
dataset_train = D2_D3_GTA_IM(datasections[:8],tag='train',
                    seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))
dataset_test = D2_D3_GTA_IM(datasections[8:10],tag='test',
                    seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))



loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle =True,
        num_workers=0,drop_last=True)

loader_val = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size, 
        shuffle =True,
        num_workers=0,drop_last=True)


#normalization 
all_train_2d = []
for cnt,batch in enumerate(loader_train): 
    if args.background_back or args.video_back:
        X,XA,y,scene = batch
    else:
        X,XA,y = batch

    all_train_2d.extend(X.flatten().numpy())
    
all_train_2d = np.asarray(all_train_2d)
_mean  = all_train_2d.mean()
_std = all_train_2d.std()

#vision normalization
if args.background_back:
    v_mean = [0.485, 0.456, 0.406]
    v_std = [0.229, 0.224, 0.225]
elif args.video_back:
    v_mean = [0.43216, 0.394666, 0.37645]
    v_std = [0.22803, 0.22145, 0.216989] 
    


print('Creating the model ....')
model = SkeletonGraph(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
                 in_channels=args.input_size,out_channels=args.output_size,
                 seq_len=args.obs_seq_len,pred_seq_len=args.pred_seq_len,kernel_size=3,
                   learn_A=args.learn_A,video_back=args.video_back,background_back=args.background_back).cuda()


checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    
print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


#load model trained 
model.load_state_dict(torch.load(checkpoint_dir+'val_mpjpe_best.pth'))

def create_skeleton_viz_data(nskeletons, njoints, flag):
    #nskeletons: number of frames
    #njoints: number of skeleton points
    lines = []
    colors = []
    for i in range(nskeletons):
        #generating line connecting the skeleton points connected in Graph
        cur_lines = np.asarray(LIMBS)
        cur_lines += i * njoints
        lines.append(cur_lines)

        single_color = np.zeros([njoints, 3])
        if flag == 1:
            # different color for target and prediction
            single_color[:] = [1.0, float(i) / nskeletons, 0.0]
        else:
            single_color[:] = [0.0, float(i) / nskeletons, 0]
        colors.append(single_color[1:])

    lines = np.concatenate(lines, axis=0)
    colors = np.asarray(colors).reshape(-1, 3)
    return lines, colors

Predictions = []
Groundtrues = []
for cnt,batch in enumerate(loader_val): 
    #get prediction and target and store them in list
    if args.background_back:
        X,XA,y,scene = batch
        scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
        scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
        scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
        scene =scene[:,0,...].cuda()
    elif args.video_back:
        X,XA,y,scene = batch
        scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
        scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
        scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
        scene =scene.cuda()
        scene = scene.transpose(1,2)
    else: 
        X,XA,y = batch

    X =(X-_mean)/_std
    X,XA,y = X.cuda(),XA.cuda(),y.cuda()
    if args.background_back or args.video_back:
        V_pred = model(X,XA,scene)
    else:
        V_pred = model(X,XA)
    Predictions.append(V_pred)
    Groundtrues.append(y)

print(len(Predictions)) 
print(len(Groundtrues))


batch_id = 18
prediction = Predictions[batch_id]
groundtrue = Groundtrues[batch_id]


for sid in range(1,128):
    predict_in_sample = prediction[sid]
    target_in_sample = groundtrue[sid]
    joints = predict_in_sample.cpu().detach().numpy()
    joints_target = target_in_sample.cpu().detach().numpy()
    tl, jn, _ = joints.shape
    # because the GTA dataset only output 10 frames so all the points in each frame are close, we need to add offset
    for i in range(1,tl):
        joints[i,:,2] = joints[i,:,2] + 0.5*i
  
    for i in range(1,tl):
        joints_target[i,:,2] = joints_target[i,:,2] + 0.5*i

    np.save('visualization/joints_'+str(sid) + '.npy',joints)
    np.save('visualization/joints_target_' + str(sid)+'.npy',joints_target)
    joints = joints.reshape(-1, 3)
    joints_target = joints_target.reshape(-1, 3)
    nskeletons = tl
    
    #prediction drawing
    lines, colors = create_skeleton_viz_data(nskeletons, jn, 1)
    #use open3d to draw geometry
    line_set = o3d.geometry.LineSet()   #create blank lineset
    line_set.points = o3d.utility.Vector3dVector(joints)     #generate point objects
    line_set.lines = o3d.utility.Vector2iVector(lines)      #generate line objects
    line_set.colors = o3d.utility.Vector3dVector(colors)     #generate color property     
    o3d.io.write_line_set('visualization/lineset_' +str(sid) + '.ply',line_set)
    
    #target drawing
    lines_target, colors_target = create_skeleton_viz_data(nskeletons, jn, 0)
    line_set_target = o3d.geometry.LineSet()
    line_set_target.points = o3d.utility.Vector3dVector(joints_target)
    line_set_target.lines = o3d.utility.Vector2iVector(lines_target)
    line_set_target.colors = o3d.utility.Vector3dVector(colors_target)
    o3d.io.write_line_set('visualization/lineset_target_' +str(sid) + '.ply',line_set_target)
    
# visualization    
line_set = o3d.io.read_line_set('visualization/lineset_63.ply')
line_set_target = o3d.io.read_line_set('visualization/lineset_target_63.ply')
vis_list = []
vis_list.append(line_set)
vis_list.append(line_set_target)

joints = np.load('visualization/joints_63.npy')
joints_target = np.load('visualization/joints_target_63.npy')
tl, jn, _ = joints.shape
print(joints.shape)
nskeletons = tl
joints = joints.reshape(-1, 3)
joints_target = joints_target.reshape(-1, 3)
for j in range(joints.shape[0]):
        # spine joints
        if j % jn == 11 or j % jn == 12 or j % jn == 13:
            continue
        transformation1 = np.identity(4)
        transformation2 = np.identity(4)
        transformation1[:3, 3] = joints[j]
        transformation2[:3, 3] = joints_target[j]
        # head joint
        if j % jn == 0:
            r = 0.07
        else:
            r = 0.03
        #draw spheres to represent points
        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere1.paint_uniform_color([1.0, float(j // jn) / nskeletons, 0.0])
        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        sphere2.paint_uniform_color([0.0, float(j // jn) / nskeletons, 0.0])
        vis_list.append(sphere1.transform(transformation1))
        vis_list.append(sphere2.transform(transformation2))
o3d.visualization.draw_geometries_with_custom_animation(vis_list)