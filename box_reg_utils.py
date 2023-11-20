import numpy as np
import torch

import math
import shapely.geometry
import shapely.affinity


#if gt_rz, prop_rz in bottom coord in lidar, difference between them needs to be adjusted
def find_delta(gt_rz,prop_rz): #only in loss function -- all in gpu 
            
    delta = gt_rz - prop_rz #in gpu
    ind = torch.logical_and(gt_rz > math.pi/2, prop_rz < -math.pi/2)  #delta will be > 180, adjust to 2*pi - delta, results in small +ve
    delta[ind] = torch.tensor([2*math.pi],device='cuda').repeat(torch.sum(ind)) - delta[ind]
    ind = torch.logical_and(gt_rz < -math.pi/2, prop_rz > math.pi/2)  #delta will be < -180, adjust to 2*pi + delta, results in small -ve
    delta[ind] = torch.tensor([2*math.pi],device='cuda').repeat(torch.sum(ind)) + delta[ind]

    return delta

def rollover_lidar_angle(rz,is_gpu):  #if is_gpu: i/p and o/p in gpu, else cpu

    if is_gpu:
        rz_result = rz.clone().detach() #in cpu or gpu depending on rz
        ind = rz > math.pi
        rz_result[ind] = rz[ind] - torch.tensor([2*math.pi],device='cuda').repeat(torch.sum(ind)) 
        ind = rz < -math.pi
        rz_result[ind] = rz[ind] + torch.tensor([2*math.pi],device='cuda').repeat(torch.sum(ind))

    else:
        rz_result = rz.clone().detach() #in cpu or gpu depending on rz
        ind = rz > math.pi
        rz_result[ind] = rz[ind] - torch.tensor([2*math.pi],device='cpu').repeat(torch.sum(ind)) 
        ind = rz < -math.pi
        rz_result[ind] = rz[ind] + torch.tensor([2*math.pi],device='cpu').repeat(torch.sum(ind))

    return rz_result



#gt_box_mid[0], gt_box_mid[1], gt_box_mid[2], gt_velo_evec0_dim, gt_velo_evec1_dim, gt_velo_dz, gt_velo_rz, 
#prop_m[0],prop_m[1],prop_m[2],prop_bbox_evec0_dim,prop_bbox_evec1_dim,prop_bbox_velo_dz,prop_velo_rz 
# 0        1         2         3                   4                   5                 6              
#prop_dx,dy used in place of prop_bbox[:,3,4] 


def estimate_gt_box(reg_bbox_preds, prop_bbox, is_gpu):   #if is_gpu: all i/p and o/p on gpu, else cpu

        #print('estimate_gt_box')
        B, _ = reg_bbox_preds.shape  
        if(is_gpu):  
            gt_box_est = torch.zeros([B,7],dtype=torch.float64,device='cuda')  #tensor 
        else:
            gt_box_est = torch.zeros([B,7],dtype=torch.float64,device='cpu')


        gt_box_est[:,0] = reg_bbox_preds[:,0] * prop_bbox[:,3] + prop_bbox[:,0]
        gt_box_est[:,1] = reg_bbox_preds[:,1] * prop_bbox[:,3] + prop_bbox[:,1]
        gt_box_est[:,2] = reg_bbox_preds[:,2] * prop_bbox[:,3] + prop_bbox[:,2]
        
        gt_box_est[:,3] = torch.exp(reg_bbox_preds[:,3]) * prop_bbox[:,3]
        gt_box_est[:,4] = torch.exp(reg_bbox_preds[:,4]) * prop_bbox[:,4]
        gt_box_est[:,5] = torch.exp(reg_bbox_preds[:,5]) * prop_bbox[:,5]
        
        #reg_bbox_preds[:,6] is predicted delta_tgt_rz                
        gt_box_est[:,6] = rollover_lidar_angle(reg_bbox_preds[:,6] + prop_bbox[:,6],is_gpu) 
                                                      
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



def IoU_3D(gtbox,gtbox_pred):  #shapely needs i/ps in cpu, so always in cpu  #assumes boxes rotated only around vertical 
    '''
    print('gtbox')
    print(gtbox)
    print('gtbox_pred')
    print(gtbox_pred)
    '''
    B = gtbox.shape[0]  

    gt_velo_mid_fwd = gtbox[:,0]    #lidar x
    gt_velo_mid_across = gtbox[:,1] #lidar y
    gt_velo_mid_vert = gtbox[:,2]
    gt_velo_evec0_dim = gtbox[:,3]  
    gt_velo_evec1_dim = gtbox[:,4]  
    gt_velo_dim_vert = gtbox[:,5]  #vert
    gt_velo_evec0_rz = gtbox[:,6] # angle of evec0 with lidar x;    +ve ccw upto pi,  -ve cw upto -pi, need to convert

    gtp_velo_mid_fwd = gtbox_pred[:,0]
    gtp_velo_mid_across = gtbox_pred[:,1]
    gtp_velo_mid_vert = gtbox_pred[:,2]
    gtp_velo_evec0_dim = gtbox_pred[:,3]  
    gtp_velo_evec1_dim = gtbox_pred[:,4]  
    gtp_velo_dim_vert = gtbox_pred[:,5]  #vert
    gtp_velo_evec0_rz = gtbox_pred[:,6]

    gt_velo_top    = gt_velo_mid_vert + gt_velo_dim_vert / 2
    gt_velo_bottom = gt_velo_mid_vert - gt_velo_dim_vert / 2

    gtp_velo_top    = gtp_velo_mid_vert + gtp_velo_dim_vert / 2
    gtp_velo_bottom = gtp_velo_mid_vert - gtp_velo_dim_vert / 2

    intersec_top    = torch.min(gt_velo_top,gtp_velo_top)
    intersec_bottom = torch.max(gt_velo_bottom,gtp_velo_bottom)
    
    intersec_ht  = torch.max(torch.zeros(B,dtype=torch.float64,device='cpu'), intersec_top - intersec_bottom)
   
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
    gt_vol  = gt_velo_evec0_dim  * gt_velo_evec1_dim  * gt_velo_dim_vert
    gtp_vol = gtp_velo_evec0_dim * gtp_velo_evec1_dim * gtp_velo_dim_vert
    '''
    print('gt_vol')
    print(gt_vol)
    print('gtp_vol')
    print(gtp_vol)
    '''
    
    intersec_area = torch.zeros(B,dtype=torch.float64,device='cpu')
    
    for ind in range(B):
           
        gt_rz =  gt_velo_evec0_rz[ind]        
        #print(gt_rz)

        if(gt_rz<0):  #convert to +ve ccw upto 2*pi
            gt_rz = 2*np.pi + gt_rz
           
        #print(gt_velo_mid_across[ind])  
        #print(gt_velo_mid_fwd[ind])
        #print(gt_velo_evec1_dim[ind])
        #print(gt_velo_evec0_dim[ind])
        #print(gt_velo_dim_vert[ind])
        #print(gt_rz)
        #print("\n")

        r1 = RotatedRect(gt_velo_mid_across[ind],gt_velo_mid_fwd[ind],gt_velo_evec1_dim[ind],gt_velo_evec0_dim[ind],gt_rz) #shapely needs all in cpu
                   
        gtp_rz =  gtp_velo_evec0_rz[ind]
        #print(gtp_rz)
          
        if(gtp_rz<0):
            gtp_rz = 2*np.pi + gtp_rz
        
        #print(gtp_velo_mid_across[ind])  
        #print(gtp_velo_mid_fwd[ind])
        #print(gtp_velo_evec1_dim[ind])
        #print(gtp_velo_evec0_dim[ind])
        #print(gtp_velo_dim_vert[ind])
        #print(gtp_rz)
        #print("\n")

        r2 = RotatedRect(gtp_velo_mid_across[ind],gtp_velo_mid_fwd[ind],gtp_velo_evec1_dim[ind],gtp_velo_evec0_dim[ind],gtp_rz)
         
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

