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
from gtaimdset import D2_D3_GTA_IM
from proxdset import D2_D3_PROX
from model import SkeletonGraph
import glob
import hashlib


parser = argparse.ArgumentParser()

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
original_tag = args.tag

args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only' or k =='torso_joint':
        continue
    args_hash += str(k)+str(v)
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()
args.tag+=args_hash

#dataset settings
if args.dataset == 'GTA_IM':
    datasections = glob.glob('./GTAIMFPS5/*/')
    load_img = False
    if args.video_back or args.background_back:
        load_img = True
    dataset_train = D2_D3_GTA_IM(datasections[:8],tag='train',
                        seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))
    dataset_test = D2_D3_GTA_IM(datasections[8:10],tag='test',
                        seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))
    

elif args.dataset == 'PROX':
    record_sections = glob.glob('PROX/recordings/*/')
    keyp_sections = glob.glob('PROX/keypoints/*/')
    load_img = False
    if args.video_back or args.background_back:
        load_img = True
    dataset_train = D2_D3_PROX(record_sections[:52],keyp_sections[:52],tag='train',
                        seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))
    dataset_test = D2_D3_PROX(record_sections[52:60],keyp_sections[52:60],tag='test',
                        seq_in=args.obs_seq_len, seq_out=args.pred_seq_len,load_img = load_img, load_depth =False, img_resize=(args.im_w,args.im_h))

loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle =True,
        num_workers=0,drop_last=True)

loader_val = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size, 
        shuffle =False,
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

#loss function

if args.use_cons:
#     def consistency_loss(V_pred,V_trgt):
#         cos_loss = None 
#         norm_loss = None

#         for i in range(V_pred.shape[2]):
#             for k in range(i,V_pred.shape[2]):
#                 if cos_loss is None: 
#                     cos_loss = cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) 
#                     norm_loss = torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1) 
#                     - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1)
#                 else:
#                     cos_loss += cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) 
#                     norm_loss += torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1) 
#                     - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1) 
#         return  args.l_norm * norm_loss.mean() + args.l_cos *cos_loss.mean()

    cos_sim = nn.CosineSimilarity(dim=-1).cuda()

    if args.l_norm ==0: #only cos 
        def consistency_loss(V_pred,V_trgt):
            cos_loss = None 
            cnt = 0 
            for i in range(V_pred.shape[2]):
#             for k in range(i+1,V_pred.shape[2]):
                k = (i +1) %V_pred.shape[2]
                if cos_loss is None: 
                    cos_loss =torch.abs(cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) )
                else:
                    cos_loss += torch.abs(cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) )
                cnt+=1


            cos_loss = cos_loss.sqrt()
            cos_loss=(1/cnt)*cos_loss

            return   args.l_cos *cos_loss.mean()
        
    elif  args.l_cos ==0: #only norm 
        def consistency_loss(V_pred,V_trgt):
            norm_loss = None
            cnt = 0 
            for i in range(V_pred.shape[2]):
                for k in range(i+1,V_pred.shape[2]):
                    if norm_loss is None: 
                        norm_loss = torch.abs(torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1)  - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1))
                    else:
                        norm_loss += torch.abs(torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1) - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1))

                    cnt+=1

            norm_loss/=cnt

            return  args.l_norm * norm_loss.mean() 

    else: #both norm and cos
        def consistency_loss(V_pred,V_trgt):
            cos_loss = None 
            norm_loss = None
            cnt = 0 
            for i in range(V_pred.shape[2]):
#                 for k in range(i+1,V_pred.shape[2]):
                k = (i +1) %V_pred.shape[2]
                if cos_loss is None: 
                    cos_loss =torch.abs(cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) )
                else:
                    cos_loss += torch.abs(cos_sim(V_trgt[:,:,i,:],V_trgt[:,:,k,:]) - cos_sim(V_pred[:,:,i,:],V_pred[:,:,k,:]) )
                        
            for i in range(V_pred.shape[2]):
                for k in range(i+1,V_pred.shape[2]):
                    if norm_loss is None: 
                        norm_loss = torch.abs(torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1)  - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1))
                    else:
                        norm_loss += torch.abs(torch.linalg.norm(V_trgt[:,:,i,:]-V_trgt[:,:,k,:],dim=-1) - torch.linalg.norm(V_pred[:,:,i,:]-V_pred[:,:,k,:],dim=-1))

                    cnt+=1

            norm_loss/=cnt
            cos_loss/=cnt
            cos_loss = cos_loss.sqrt()


            return  args.l_norm * norm_loss.mean() + args.l_cos *cos_loss.mean()

    l2loss = nn.MSELoss().cuda()
    def graph_loss(V_pred,V_trgt):

        return l2loss(V_pred,V_trgt)+ consistency_loss(V_pred,V_trgt)
else:
    l2loss = nn.MSELoss().cuda()
    def graph_loss(V_pred,V_trgt):

        return l2loss(V_pred,V_trgt)
    
#model 
print('Creating the model ....')
model = SkeletonGraph(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
                 in_channels=args.input_size,out_channels=args.output_size,
                 seq_len=args.obs_seq_len,pred_seq_len=args.pred_seq_len,kernel_size=3,
                   learn_A=args.learn_A,video_back=args.video_back,background_back=args.background_back).cuda()

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    
checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    
print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#train val functions
def train(epoch):
    global metrics,constant_metrics

    model.train()
    loss_train = 0 

    for cnt,batch in enumerate(loader_train): 
        #get data
        if args.background_back:
            X,XA,y,scene = batch
#             scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#             scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#             scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
            scene =scene[:,-1,...].cuda()
        elif args.video_back:
            X,XA,y,scene = batch
#             scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#             scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#             scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
            scene = scene.view(scene.shape[0],scene.shape[1]*scene.shape[2],scene.shape[3],scene.shape[4]).cuda()
#             scene = scene.transpose(1,2)
        else: 
            X,XA,y = batch

        X =(X-_mean)/_std
        X,XA,y = X.cuda(),XA.cuda(),y.cuda()
        
        #Forward, backward and optimize
        optimizer.zero_grad()
        if args.background_back or args.video_back:
            V_pred = model(X,XA,scene)
        else:
            V_pred = model(X,XA)

        loss = graph_loss(V_pred,y)
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        optimizer.step()
        
        #Logging
        loss_train += loss.item()
        if cnt%args.log_frq == 0 and cnt!=0:
            print('Epoch:', epoch,'\t Train Loss:',loss_train/(cnt+1))
            
    metrics['train_loss'].append(loss_train/(cnt+1))
    



def vald(epoch):
    global metrics,constant_metrics
    model.eval()
    loss_val = 0 
    mpjpe_avg = 0 

    with torch.no_grad(): #Faster without grad 
        for cnt,batch in enumerate(loader_val): 
            #get data
            if args.background_back:
                X,XA,y,scene = batch
#                 scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#                 scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#                 scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
                scene =scene[:,-1,...].cuda()
            elif args.video_back:
                X,XA,y,scene = batch
#                 scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#                 scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#                 scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
                scene = scene.view(scene.shape[0],scene.shape[1]*scene.shape[2],scene.shape[3],scene.shape[4]).cuda()
#                 scene = scene.transpose(1,2)
            else: 
                X,XA,y = batch

            X =(X-_mean)/_std
            X,XA,y = X.cuda(),XA.cuda(),y.cuda()

            #Forward, loss, eval metrics
            if args.background_back or args.video_back:
                V_pred = model(X,XA,scene)
            else:
                V_pred = model(X,XA)
            loss = graph_loss(V_pred,y)
            
            mpjpe = MPJPE(V_pred,y)
            #Logging
            loss_val += loss.item()
            mpjpe_avg += mpjpe.item()
            if cnt%args.log_frq == 0 and cnt!=0:
                print('Epoch:', epoch,'\t Val Loss:',loss_val/(cnt+1),'\t MPJPE:',mpjpe_avg/(cnt+1))

    metrics['val_loss'].append(loss_val/(cnt+1))
    metrics['val_mpjpe'].append(mpjpe_avg/(cnt+1))

    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_loss_best.pth')  
        
    if  metrics['val_mpjpe'][-1]< constant_metrics['min_val_mpjpe']:
        constant_metrics['min_val_mpjpe'] =  metrics['val_mpjpe'][-1]
        constant_metrics['min_val_mpjpe_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_mpjpe_best.pth')  


#Train or eval? 
if not args.eval_only:
    print('Training started ...')

    #Training 
    metrics = {'train_loss':[],  'val_loss':[], 'val_mpjpe':[]}
    constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999,
                        'min_val_mpjpe_epoch':-1, 'min_val_mpjpe':9999999999999999}

    for epoch in range(args.num_epochs):
        with torch.autograd.set_detect_anomaly(True):
            train(epoch)
        vald(epoch)
        if args.use_lrschd:
            scheduler.step()


        print('*'*30)
        print('Epoch:',args.tag,":", epoch)
        for k,v in metrics.items():
            if len(v)>0:
                print(k,v[-1])


        print(constant_metrics)
        print('*'*30)

        with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp) 
else:

    def eval_metrics():
        model.eval()
        loss_val = 0
        
        mpjpe_avg = 0 
        mpjpe_qaurt_avg =0 
        mpjpe_half_avg  =0
        mpjpe_3quart_avg =0 

        mpjpe_path_avg = 0 
        mpjpe_path_qaurt_avg =0 
        mpjpe_path_half_avg  =0
        mpjpe_path_3quart_avg =0 

        with torch.no_grad(): #Faster without grad 
            for cnt,batch in enumerate(loader_val): 
                #get data
                if args.background_back:
                    X,XA,y,scene = batch
#                     scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#                     scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#                     scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
                    scene =scene[:,0,...].cuda()
                elif args.video_back:
                    X,XA,y,scene = batch
#                     scene[...,0] = (scene[...,0]-v_mean[0])/v_std[0]
#                     scene[...,1] = (scene[...,1]-v_mean[1])/v_std[1]
#                     scene[...,2] = (scene[...,2]-v_mean[2])/v_std[2]
                    scene = scene.view(scene.shape[0],scene.shape[1]*scene.shape[2],scene.shape[3],scene.shape[4]).cuda()
#                     print(scene.shape)#                     print(scene.shape)
                else: 
                    X,XA,y = batch

                X =(X-_mean)/_std
                X,XA,y = X.cuda(),XA.cuda(),y.cuda()

                #Forward, loss, eval metrics
                if args.background_back or args.video_back:
                    V_pred = model(X,XA,scene)
                else:
                    V_pred = model(X,XA)
                    
                loss_val += graph_loss(V_pred,y).item()
                
                mpjpe_avg+= MPJPE(V_pred,y).item()
                mpjpe_qaurt_avg+=MPJPE_timelimit(V_pred,y,0.25).item()
                mpjpe_half_avg+=MPJPE_timelimit(V_pred,y,0.50).item()
                mpjpe_3quart_avg+=MPJPE_timelimit(V_pred,y,0.75).item()
                
                mpjpe_path_avg+= MPJPE_torso(V_pred,y,torso_joint=args.torso_joint).item() 
                mpjpe_path_qaurt_avg+=MPJPE_torso_timelimit(V_pred,y,0.25,torso_joint=args.torso_joint).item()  
                mpjpe_path_half_avg+=MPJPE_torso_timelimit(V_pred,y,0.50,torso_joint=args.torso_joint).item() 
                mpjpe_path_3quart_avg+=MPJPE_torso_timelimit(V_pred,y,0.75,torso_joint=args.torso_joint).item()  
                
        loss_val /= (cnt+1)
        mpjpe_avg /= (cnt+1)
        mpjpe_qaurt_avg /= (cnt+1)
        mpjpe_half_avg /= (cnt+1)
        mpjpe_3quart_avg /= (cnt+1)

        mpjpe_path_avg /= (cnt+1)
        mpjpe_path_qaurt_avg /= (cnt+1)
        mpjpe_path_half_avg  /= (cnt+1)
        mpjpe_path_3quart_avg /= (cnt+1)
        
        
        mpjpe_avg = int(mpjpe_avg*1000)
        mpjpe_qaurt_avg = int(mpjpe_qaurt_avg*1000)
        mpjpe_half_avg = int(mpjpe_half_avg*1000)
        mpjpe_3quart_avg = int(mpjpe_3quart_avg*1000)

        mpjpe_path_avg = int(mpjpe_path_avg*1000)
        mpjpe_path_qaurt_avg = int(mpjpe_path_qaurt_avg*1000)
        mpjpe_path_half_avg  =int(mpjpe_path_half_avg*1000)
        mpjpe_path_3quart_avg = int(mpjpe_path_3quart_avg*1000)
        
        FDE = int((mpjpe_avg+mpjpe_path_avg)/2)
        ADE = int(((mpjpe_avg+mpjpe_qaurt_avg+mpjpe_half_avg+mpjpe_3quart_avg)/4 + (mpjpe_path_avg+mpjpe_path_qaurt_avg+mpjpe_path_half_avg+mpjpe_path_3quart_avg)/4)/2)
        
        var_pose = np.var(np.array([mpjpe_avg, mpjpe_qaurt_avg,mpjpe_half_avg,mpjpe_3quart_avg]))
        var_path = np.var(np.array([mpjpe_path_avg, mpjpe_path_qaurt_avg,mpjpe_path_half_avg,mpjpe_path_3quart_avg]))
        STB = int(np.sqrt(0.5*var_pose+0.5*var_path))

        print('#'*30)
        print('All results are in mm')
        print('*'*30)
        print('MPJPE POSE: 0.25\t 0.50\t 0.75\t full')
        print('MPJPE: ',mpjpe_qaurt_avg,'\t ',mpjpe_half_avg,'\t ',mpjpe_3quart_avg,'\t ',mpjpe_avg,'')
        print('*'*30)
        print('MPJPE PATH: 0.25\t 0.50\t 0.75\t full')
        print('PATH: ',mpjpe_path_qaurt_avg,'\t ',mpjpe_path_half_avg,'\t ',mpjpe_path_3quart_avg,'\t ',mpjpe_path_avg,'')
        print('#'*30)
        print('#'*30)
        print('FDE:',FDE)
        print('ADE:',ADE)
        print('STB:',STB)

        print('#'*30)
        print('#'*30)

        f = open(checkpoint_dir+"eval.csv", "w")
        eval_id = ''
        for k,v in vars(args).items():
            if k == 'eval_only' or k =='torso_joint' or k =='tag':
                continue
            eval_id += str(k)+str(v)
        eval_result = [eval_id,',',original_tag,',',mpjpe_path_qaurt_avg,',',mpjpe_path_half_avg,',',mpjpe_path_3quart_avg,',',mpjpe_path_avg,',',mpjpe_qaurt_avg,',',mpjpe_half_avg,',',mpjpe_3quart_avg,',',mpjpe_avg,',',FDE,',',ADE,',',STB,'\n']
        eval_result_row = ''
        for ss in eval_result:
            eval_result_row+= str(ss)
        f.write(eval_result_row)
        f.close()
    #Eval at differnet time steps and 3d pose, 3d positions
    #Load the model weights 
    model.load_state_dict(torch.load(checkpoint_dir+'val_mpjpe_best.pth'))
    eval_metrics()
