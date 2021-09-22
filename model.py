#Define the model 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class SpatioTemporalGraphCNN(nn.Module):

    def __init__(self,
                 in_channels=2,
                 out_channels=3, #The parameters of the distribution = 10*3
                 seq_in = 8,
                 kernel_size=3,learn_A=True):
        super(SpatioTemporalGraphCNN,self).__init__()
        
        self.learn_A = learn_A
        
        self.cnn = nn.Conv2d(in_channels,out_channels,(1,kernel_size))
        
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(seq_in),
            nn.PReLU(),
            nn.Conv2d(
                seq_in,
                seq_in,
                kernel_size,
                padding=1,
                padding_mode='replicate' #This effects the results 
            ),
            nn.BatchNorm2d(seq_in),
            nn.Dropout(0, inplace=True),
        )
        
        #Learning A 
        if self.learn_A:
            self.extractCNN = nn.Sequential(
                nn.BatchNorm2d(seq_in),
                nn.PReLU(),
                nn.Conv2d(
                    seq_in,
                    seq_in,
                    (3,3),
                    padding=(1,1),
                    padding_mode='replicate' #This effects the results 
                ),
                nn.BatchNorm2d(seq_in),
            )
        
    def forward(self,x,A): #x = [batch,seq_in,joints,x|y],A= [batch,seq_in,joints,joints]
        
        if self.learn_A:
            A = self.extractCNN(A)
        
        
        x = x.transpose(1,3) #[batch,x|y,joints,seq_in]
        x = self.cnn(x)
        x = torch.einsum('bfjy,bsjv->bfjs',(x,A)) # [batch,distr features,joints,seq_in]
        x = x.transpose(1,3) #[batch,seq_in,distr features,joints]
        
        x = self.tcn(x)+x #[batch,seq_in,distr features,joints]
        
        return x,A 
        
class SkeletonGraph(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=3,
                 in_channels=2,out_channels=3,
                 seq_len=8,pred_seq_len=12,kernel_size=3,learn_A=True,video_back=False,background_back=False):
        super(SkeletonGraph,self).__init__()
        
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.video_back = video_back
        self.background_back = background_back
        
        with_vis = 0 
        if video_back or background_back:
            with_vis=1
        
        #Spatio-Temporal Embedding
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(SpatioTemporalGraphCNN(
                 in_channels=in_channels,
                 out_channels=out_channels, 
                 seq_in = seq_len,
                 kernel_size=kernel_size,learn_A=learn_A))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(SpatioTemporalGraphCNN(
                 in_channels=out_channels,
                 out_channels=out_channels, 
                 seq_in = seq_len,
                 kernel_size=kernel_size,learn_A=learn_A))

            
        #Time Extrapolater 
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Sequential(nn.Conv2d(seq_len+with_vis,pred_seq_len,kernel_size,padding=1),
                                          nn.BatchNorm2d(pred_seq_len) ) )
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Sequential(nn.Conv2d(pred_seq_len,pred_seq_len,kernel_size,padding=1),
                                          nn.BatchNorm2d(pred_seq_len) ))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,kernel_size,padding=1)
        
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
            
        if self.video_back:
            self.vision_unit = nn.Sequential(nn.Conv2d(3*seq_len,6,(3,3),stride=(1,1),padding_mode='replicate'),nn.BatchNorm2d(6),nn.PReLU(),
                            nn.Conv2d(6,9,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(9),nn.PReLU(),
                            nn.Conv2d(9,12,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(12),nn.PReLU(),
                            nn.Conv2d(12,15,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(15),nn.PReLU(),
                            nn.Conv2d(15,18,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(18),nn.PReLU(),
                            nn.Conv2d(18,21,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(21),nn.PReLU())


 
        if self.background_back:
            self.vision_unit = nn.Sequential(nn.Conv2d(3,6,(3,3),stride=(1,1),padding_mode='replicate'),nn.BatchNorm2d(6),nn.PReLU(),
                            nn.Conv2d(6,9,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(9),nn.PReLU(),
                            nn.Conv2d(9,12,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(12),nn.PReLU(),
                            nn.Conv2d(12,15,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(15),nn.PReLU(),
                            nn.Conv2d(15,18,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(18),nn.PReLU(),
                            nn.Conv2d(18,21,(3,3),stride=(2,2),padding_mode='replicate'),nn.BatchNorm2d(21),nn.PReLU())


        
    def forward(self,v,a,scene=None): #v = [batch,seq_in,joints,x|y],a= [batch,seq_in,joints,joints]

        v,a = self.st_gcns[0](v,a)

        for k in range(1,self.n_stgcnn):
            v_,a_ = self.st_gcns[k](v,a)
            v = v_+v
            a = a_+a

            
        if self.background_back or self.video_back:
#             print(scene.shape)
            vis = self.vision_unit(scene).permute(0,3,1,2) #torch.Size([64, 21, 3, 1])
            v = torch.cat([v,vis],dim=1)

        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        
        
        return v  #[batch,seq_out,joints,dist fetaures]  
        
        
        

#Build the spation-temporal_graph_batched_cnn =( ) don/t forget to have the the V pass on conv first,then
#einsum this thing 
#next add the temporal dimension 
#later use the embedding to predict next steps 
#ucan have the A learnable (might need to normalize it .)
