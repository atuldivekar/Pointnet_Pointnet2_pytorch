import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from pointnet2_utils import PointNetSetAbstraction
import numpy as np
import box_reg_utils

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, num_class)        
        self.bbox_pred = nn.Linear(256, 7) # curr model, class --> predict tx, ty tz tw th tl tgt_rz
            

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x1 = self.drop2(F.relu(self.bn2(self.fc2(x))))  #feature vector feeds 2 heads

        x = self.fc3(x1) #ip is B * 256, op B * num_class -- logits of being in each class
        class_pred = F.log_softmax(x, -1)  #op B * num_class  -- log(prob) of being in each class
        
        reg_bbox_preds = self.bbox_pred(x1) #op B * 7
                
        return class_pred, reg_bbox_preds
        
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def smooth_l1_loss_single_arg(self, input, beta=1. / 9, reduction="mean"):
        
        n = torch.abs(input)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if reduction=="mean":
          return loss.mean()
        return loss.sum()

    def smooth_l1_loss(self, input, target, beta=1. / 9, reduction="mean"):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        return self.smooth_l1_loss_single_arg(input-target, beta=beta, reduction=reduction)

#gt_box_mid[0], gt_box_mid[1], gt_box_mid[2], gt_velo_evec0_dim, gt_velo_evec1_dim, gt_velo_dz, gt_evec0_rz
#prop_m[0],prop_m[1],prop_m[2],prop_bbox_evec0_dim,prop_bbox_evec1_dim,prop_bbox_velo_dz,prop_evec0_rz 
# 0        1         2           3                   4                   5                 6             

    #uses current batch gt_bbox and prop_box to get tgt deltas
    #model gives predicted deltas
    #loss is difference of the two
    #backprop finds how to adjust weights to reduce this loss
    
    
    def forward(self, class_pred, class_target, reg_bbox_pred, prop_bbox, gt_bbox):  #all in gpu

        #print('loss')
        class_loss = F.nll_loss(class_pred, class_target) #class_pred B * num_class Tensor in GPU  -- log(prob) of being in each class
                                                          #class tgt B * 1 -- true class index 
        #print('class_loss')
        #print(class_loss)
           
        #gt_bbox prop_bbox are Tensors in gpu as needed

        tgt_tx = (gt_bbox[:,0]-prop_bbox[:,0])/prop_bbox[:,3]  #mid[0] each (B,) 
        tgt_ty = (gt_bbox[:,1]-prop_bbox[:,1])/prop_bbox[:,3]  #mid[1] prop_bbox_evec0_dim to scale all - always nonzero, has dimensions comparable to numerator
        tgt_tz = (gt_bbox[:,2]-prop_bbox[:,2])/prop_bbox[:,3]  #mid[2] cannot use tgt_tx = gtbox[:,0] since clusters are mean subtracted --do not carry location info.
        tgt_tdx = torch.log(gt_bbox[:,3]/prop_bbox[:,3])       #evec0_dim 
        tgt_tdy = torch.log(gt_bbox[:,4]/prop_bbox[:,4])       #evec1_dim 
        tgt_tdz = torch.log(gt_bbox[:,5]/prop_bbox[:,5])       #height
    
        tgt = torch.stack((tgt_tx, tgt_ty, tgt_tz, tgt_tdx, tgt_tdy, tgt_tdz),axis=1) #all except delta_rz

        tgt_delta_rz =  box_reg_utils.find_delta(gt_bbox[:,6],prop_bbox[:,6])  #reg tgt is gt_evec0_rz - prop_evec0_rz  ; all in gpu

        #print('tgt')
        #print(tgt)        
        #print(tgt_delta_rz)

        #tgt = from_numpy(tgt).cuda().float()   #smooth_l1_loss takes tensors not np array, does not take long.              
        reg_box_p = reg_bbox_pred[:,0:6]       
        box_loss = self.smooth_l1_loss(reg_box_p.float(), tgt,  beta =  1. / 9, reduction="mean")

        #print('box_loss')
        #print(box_loss)
               
        ang_err = torch.sin(tgt_delta_rz - reg_bbox_pred[:,6])
        #tgt_delta_rz - reg_bbox_pred[:,6]  = gt_rz - proposal_rz - gtp_rz + proposal_rz_pred

        ang_loss =  self.smooth_l1_loss_single_arg(ang_err, beta = 1. / 9, reduction="mean")

        #print('ang_err')
        #print(ang_err)

        #print('ang_loss')
        #print(ang_loss)
        #input()
       
        #total_loss = class_loss  + 2.0 * (box_loss + ang_loss)  
        
        total_loss = box_loss + ang_loss

        #print('total_loss')   
        #print(total_loss.item())   
        #input()
        return total_loss

