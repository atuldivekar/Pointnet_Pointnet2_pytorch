import numpy as np
import torch

import math
import shapely.geometry
import shapely.affinity



def get_lidar_angle_no_ry(vec):   #vec in gpu

    B, S = vec.shape    
    angle = torch.empty([B],dtype = torch.double,device='cuda')  #in gpu
    abs_sine = torch.abs(vec[:,1])/torch.norm(vec,dim=1)

    ind = torch.logical_and(vec[:,0] >= 0,vec[:,1] <= 0)
    angle[ind] = -torch.asin(abs_sine[ind])

    ind = torch.logical_and(vec[:,0] < 0,vec[:,1] <= 0)        
    angle[ind] = torch.asin(abs_sine[ind]) - torch.tensor([math.pi]).repeat(torch.sum(ind)).cuda()

    ind = torch.logical_and(vec[:,0] >= 0,vec[:,1] > 0)    
    angle[ind] = torch.asin(abs_sine[ind]) 

    ind = torch.logical_and(vec[:,0] < 0,vec[:,1] > 0) 
    angle[ind] = torch.tensor([math.pi]).repeat(torch.sum(ind)).cuda() - torch.asin(abs_sine[ind]) 
 
    #print('angle')
    #print(angle)

    return angle
        

#gt_box_mid[0], gt_box_mid[1], gt_box_mid[2], gt_velo_dx, gt_velo_dy, gt_velo_dz, gt_velo_rz, 
#prop_m[0],prop_m[1],prop_m[2],prop_bbox_evec0_dim,prop_bbox_evec1_dim,prop_bbox_velo_dz,prop_velo_rz,evec[0,0],evec[1,0],evec[0,1],evec[1,1],ind_min  
# 0        1         2         3                   4                   5                 6              7         8         9         10        11
#prop_dx,dy used in place of prop_bbox[:,3,4] 


def estimate_gt_box(reg_bbox_preds, orien_pred, prop_bbox):   #all tensors in gpu returns in gpu

        #print('estimate_gt_box')
        B, _ = reg_bbox_preds.shape    
        gt_box_est = torch.zeros([B,7],dtype=torch.float64,device='cuda')  #tensor in gpu

        orien_choice = orien_pred.data.max(1)[1]  #B*1 tensor
        #print('orien_choice')
        #print(orien_choice)    
        
        use0 = torch.logical_or(orien_choice==0, orien_choice==2)
             
        #use orien_choice to select dx and dy
        prop_dx = torch.where(use0, prop_bbox[:,3], prop_bbox[:,4])
        prop_dy = torch.where(use0, prop_bbox[:,4], prop_bbox[:,3])
        
        gt_box_est[:,0] = reg_bbox_preds[:,0] * prop_dx + prop_bbox[:,0]
        gt_box_est[:,1] = reg_bbox_preds[:,1] * prop_dy + prop_bbox[:,1]
        gt_box_est[:,2] = reg_bbox_preds[:,2] * prop_bbox[:,5] + prop_bbox[:,2]
        
        gt_box_est[:,3] = torch.exp(reg_bbox_preds[:,3]) * prop_dx
        gt_box_est[:,4] = torch.exp(reg_bbox_preds[:,4]) * prop_dy
        gt_box_est[:,5] = torch.exp(reg_bbox_preds[:,5]) * prop_bbox[:,5]

        
        use0 = torch.stack([use0,use0],dim=1)        
        #print(use0)      
        #print(prop_bbox[:,7:9])
        #print(prop_bbox[:,9:11])
        #use orien_choice to select evec and find prop_rz     
        evec = torch.where(use0,prop_bbox[:,7:9],prop_bbox[:,9:11])
        #print(evec)      

        no_sign_change = torch.logical_or(orien_choice==0,orien_choice==1)
        no_sign_change = torch.stack([no_sign_change,no_sign_change],dim=1)       
        #print(no_sign_change)
        evec = torch.where(no_sign_change,evec,-evec) 
        
        #print('evec')        
        #print(evec)        
      
        prop_rz = get_lidar_angle_no_ry(evec) #evec is Tensor in gpu

        #print('prop_rz')        
        #print(prop_rz)       
        
        #reg_bbox_preds[:,6] is predicted delta_tgt_rz
        
        gt_box_est[:,6] = torch.clamp(reg_bbox_preds[:,6] + prop_rz, min=-math.pi+0.001, max=math.pi-0.001)    #if > pi or < -pi -- for now just clamp -- could rollover
                                                               
        #print('gt_box_est')
        #print(gt_box_est)                


        return gt_box_est


class RotatedRect:
    def __init__(self, cx, cy, w, h, angle): #cx horiz cy ver w hor h vert on screen
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle, use_radians=True)   #+ve ccw 0 to 2pi
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())
    
    def area(self):
         return self.w*self.h



def IoU_3D(gtbox,gtbox_pred):  #tensors in gpu  #assumes boxes rotated only around vertical 

    '''
    print('gtbox')
    print(gtbox)
    print('gtbox_pred')
    print(gtbox_pred)
    '''
    
    B = gtbox.shape[0]  

    gt_velo_mid_fwd = gtbox[:,0]
    gt_velo_mid_across = gtbox[:,1]
    gt_velo_mid_vert = gtbox[:,2]
    gt_velo_dim_fwd = gtbox[:,3]  #fwd before rot
    gt_velo_dim_across = gtbox[:,4]  #across
    gt_velo_dim_vert = gtbox[:,5]  #vert
    gt_velo_rz = gtbox[:,6] # +ve ccw upto pi,  -ve cw upto -pi, need to convert

    gtp_velo_mid_fwd = gtbox_pred[:,0]
    gtp_velo_mid_across = gtbox_pred[:,1]
    gtp_velo_mid_vert = gtbox_pred[:,2]
    gtp_velo_dim_fwd = gtbox_pred[:,3]  #fwd before rot
    gtp_velo_dim_across = gtbox_pred[:,4]  #across
    gtp_velo_dim_vert = gtbox_pred[:,5]  #vert
    gtp_velo_rz = gtbox_pred[:,6]

    gt_velo_top    = gt_velo_mid_vert + gt_velo_dim_vert / 2
    gt_velo_bottom = gt_velo_mid_vert - gt_velo_dim_vert / 2

    gtp_velo_top    = gtp_velo_mid_vert + gtp_velo_dim_vert / 2
    gtp_velo_bottom = gtp_velo_mid_vert - gtp_velo_dim_vert / 2

    intersec_top    = torch.min(gt_velo_top,gtp_velo_top)
    intersec_bottom = torch.max(gt_velo_bottom,gtp_velo_bottom)
    intersec_ht     = torch.max(torch.zeros(B,dtype=torch.float64,device='cuda'), intersec_top - intersec_bottom)
    '''
    print('gt_velo_top,gt_velo_bottom')

    print(gt_velo_top)
    print(gt_velo_bottom)

    print('gtp_velo_top,gtp_velo_bottom')
    print(gtp_velo_top)
    print(gtp_velo_bottom)
    print('intersec_ht')
    print(intersec_ht)
    '''
    gt_vol  = gt_velo_dim_fwd  * gt_velo_dim_across  * gt_velo_dim_vert
    gtp_vol = gtp_velo_dim_fwd * gtp_velo_dim_across * gtp_velo_dim_vert
    '''
    print('gt_vol')
    print(gt_vol)
    print('gtp_vol')
    print(gtp_vol)
    '''
    intersec_area = torch.zeros(B,dtype=torch.float64,device='cuda')
    
    for ind in range(B):
           
        gt_rz =  gt_velo_rz[ind]        
        #print(gt_rz)

        if(gt_rz<0):  #convert to +ve ccw upto 2*pi
            gt_rz = 2*np.pi + gt_rz
           
        #print(gt_velo_mid_across[ind])  
        #print(gt_velo_mid_fwd[ind])
        #print(gt_velo_dim_across[ind])
        #print(gt_velo_dim_fwd[ind])
        #print(gt_velo_dim_vert[ind])
        #print(gt_rz)
        #print("\n")

        r1 = RotatedRect(gt_velo_mid_across[ind],gt_velo_mid_fwd[ind],gt_velo_dim_across[ind],gt_velo_dim_fwd[ind],gt_rz) 
                   
        gtp_rz =  gtp_velo_rz[ind]
        #print(gtp_rz)
          
        if(gtp_rz<0):
            gtp_rz = 2*np.pi + gtp_rz
        
        #print(gtp_velo_mid_across[ind])  
        #print(gtp_velo_mid_fwd[ind])
        #print(gtp_velo_dim_across[ind])
        #print(gtp_velo_dim_fwd[ind])
        #print(gtp_velo_dim_vert[ind])
        #print(gtp_rz)
        #print("\n")

        r2 = RotatedRect(gtp_velo_mid_across[ind],gtp_velo_mid_fwd[ind],gtp_velo_dim_across[ind],gtp_velo_dim_fwd[ind],gtp_rz)
         
        intersec = r1.intersection(r2)

        intersec_area[ind] = intersec.area

    intersec_vol = intersec_area *  intersec_ht
    IoU3D = intersec_vol / (gt_vol + gtp_vol - intersec_vol)
    '''
    print('intersec_area')   
    print(intersec_area) 
    print('intersec_vol')
    print(intersec_vol)
    '''
    #print('IoU3D')
    #print(IoU3D)
    #input()


    return IoU3D

