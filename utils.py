import torch
import math
#Eval functions , from paper     section 5.2 they use
#Mean Per Joint Position Error (MPJPE) 
#Why this is important, because we are training a generative model in which 
#the loss might not be the best metric to choose the best eval point 
#thus we evaluate this on this metric on each epoch to pick the best weights 
#https://github.com/cbsudux/Human-Pose-Estimation-101#mean-per-joint-position-error---mpjpe 

#Mean Per Joint Position Error - MPJPE

#     Per joint position error = Euclidean distance between ground truth and prediction for a joint
#     Mean per joint position error = Mean of per joint position error for all k joints (Typically, k = 16)
#     Calculated after aligning the root joints (typically the pelvis) of the estimated and groundtruth 3D pose.


# def MPJPE(V_pred,V_trgt):  #V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}

#     B = V_pred.shape[0]
#     T = V_pred.shape[1]
#     J = V_pred.shape[2]
#     metric =0 
#     for b in range(B):
#         for t in range(T):
#             for j in range(J):
#                 metric += torch.sqrt((V_trgt[b,t,j,0]- V_pred[b,t,j,0])**2+
#                         (V_trgt[b,t,j,1]- V_pred[b,t,j,1])**2+
#                         (V_trgt[b,t,j,2]- V_pred[b,t,j,2])**2   )

#     return metric/(b*t*j)
#423 ms ± 12.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

def MPJPE(V_pred,V_trgt):#V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}
    return torch.linalg.norm(V_trgt- V_pred,dim=-1).mean() 
#1.61 ms ± 2.24 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


def MPJPE_torso(V_pred,V_trgt,torso_joint = 13):  #V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z}, torso_joint= for specific path

    return torch.linalg.norm(V_trgt[:,:,torso_joint,:]- V_pred[:,:,torso_joint,:],dim=-1).mean() 



def MPJPE_timelimit(V_pred,V_trgt,time_limit): #V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME,,time_limit=where to evaluate in time, Points,{x,y,z}
    T = V_pred.shape[1]
    T = max(int(T*time_limit),1)
    return torch.linalg.norm(V_trgt[:,:T,...]- V_pred[:,:T,...],dim=-1).mean()

def MPJPE_torso_timelimit(V_pred,V_trgt,time_limit,torso_joint = 13 ):  #V_pred = Batch,TIME, Points,{x,y,z},#V_trgt  = Batch,TIME, Points,{x,y,z},time_limit=where to evaluate in time, torso_joint= for specific path
    T = V_pred.shape[1]
    T = max(int(T*time_limit),1)
    #13 is the center of the torso in GTA IM
    return torch.linalg.norm(V_trgt[:,:T,torso_joint,:]- V_pred[:,:T,torso_joint,:],dim=-1).mean() 
#Test
# V_trgt = torch.rand(32,10,21,3) #Batch,TIME, Points,{x,y,z}
# V_pred = torch.randn(32,10,21,3) #Batch,TIME, G parameters
# print(MPJPE(V_pred,V_trgt),MPJPE_timelimit(V_pred,V_trgt,time_limit=0.25))
# print(MPJPE_timelimit(V_pred,V_trgt,time_limit=0.50),MPJPE_timelimit(V_pred,V_trgt,time_limit=0.75))
# MPJPE_torso_timelimit(V_pred,V_trgt,time_limit=0.25),MPJPE_torso(V_pred,V_trgt)
