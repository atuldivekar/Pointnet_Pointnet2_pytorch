import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from pointnet2_utils import  PointNetSetAbstractionMsg, PointNetSetAbstraction
import numpy as np
import box_reg_utils



class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        
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
        
        x = self.fc3(x1)     #ip is B * 256, op B * num_class -- logits of being in each class
        class_pred  = F.log_softmax(x, -1)  #op B * num_class  -- log(prob) of being in each class

        reg_bbox_preds = self.bbox_pred(x1) #op B * 7

        return class_pred, reg_bbox_preds

       
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def smooth_l1_loss_single_arg(self, input, beta=1. / 9, reduction="sum"):
        
        n = torch.abs(input)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if reduction=="mean":
          return loss.mean()
        return loss.sum()

    def smooth_l1_loss(self, input, target, beta=1. / 9, reduction="sum"):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        return self.smooth_l1_loss_single_arg(input-target, beta=beta, reduction=reduction)

#gt_box_mid[0], gt_box_mid[1], gt_box_mid[2], gt_velo_dx, gt_velo_dy, gt_velo_dz, gt_velo_rz
#prop_m[0],prop_m[1],prop_m[2],prop_bbox_shorter,prop_bbox_longer,prop_bbox_velo_dz,prop_shorter_rz 
# 0        1         2           3                   4                   5                 6             

    #uses current batch gt_bbox and prop_box to get tgt deltas
    #model gives predicted deltas
    #loss is difference of the two
    #backprop finds how to adjust weights to reduce this loss
    
    
    def forward(self, class_pred, class_target, reg_bbox_pred, prop_bbox, gt_bbox):  #all in gpu

        B, _ = reg_bbox_pred.shape
      
        #print('loss')
        class_loss = F.nll_loss(class_pred, class_target) #class_pred B * num_class Tensor in GPU  -- log(prob) of being in each class
                                                          #class tgt B * 1 -- true class index 
        #print('class_loss')
        #print(class_loss)
           
        #gt_bbox prop_bbox are Tensors in gpu as needed

        tgt_tx = (gt_bbox[:,0]-prop_bbox[:,0])/prop_bbox[:,4]  #mid[0] each (B,) 
        tgt_ty = (gt_bbox[:,1]-prop_bbox[:,1])/prop_bbox[:,4]  #mid[1] prop_bbox_longer to scale both tx,ty -- more stable than shorter - always nonzero, has dimensions comparable to numerator
        tgt_tz = (gt_bbox[:,2]-prop_bbox[:,2])/prop_bbox[:,5]  #mid[2] cannot use tgt_tx = gtbox[:,0] since clusters are mean subtracted --do not carry location info.
        tgt_tdx = torch.log(gt_bbox[:,3]/prop_bbox[:,3])       #w/shorter 
        tgt_tdy = torch.log(gt_bbox[:,4]/prop_bbox[:,4])       #l/longer
        tgt_tdz = torch.log(gt_bbox[:,5]/prop_bbox[:,5])       #height
    
        tgt = torch.stack((tgt_tx, tgt_ty, tgt_tz, tgt_tdx, tgt_tdy, tgt_tdz),axis=1) #all except delta_rz
        #tgt = torch.zeros([B,6],dtype=torch.float64,device='cuda')
        
        tgt_delta_rz =  gt_bbox[:,6] - prop_bbox[:,6]  #reg tgt is gt_velo_rz - prop_shorter_rz  ; all in gpu
        #tgt_delta_rz = torch.zeros(B,dtype=torch.float64,device='cuda')  #implies prop_box = gt_bbox; set this in IoU
        #print('tgt')
        #print(tgt)        
        #print(tgt_delta_rz)

        #smooth_l1_loss takes tensors not np array, does not take long.              
        reg_box_p = reg_bbox_pred[:,0:6]       
        box_loss = self.smooth_l1_loss(reg_box_p.float(), tgt,  beta =  1. / 9, reduction="sum")

        #print('box_loss')
        #print(box_loss)
               
        ang_err = torch.sin(tgt_delta_rz - reg_bbox_pred[:,6])
        #tgt_delta_rz - reg_bbox_pred[:,6]  = gt_rz - proposal_rz - gtp_rz + proposal_rz_pred

        ang_loss =  self.smooth_l1_loss_single_arg(ang_err, beta = 1. / 9, reduction="sum")

        #print('ang_err')
        #print(ang_err)

        #print('ang_loss')
        #print(ang_loss)
        #input()
       
        total_loss = class_loss  + 5.0 * (box_loss + ang_loss)  
        
        #print('total_loss')   
        #print(total_loss.item())   
        #input()
        return total_loss




