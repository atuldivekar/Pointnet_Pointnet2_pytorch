'''
@author: Atul Divekar
@file: KittiAdbscanDataLoader.py
@time: 2023/5/15
'''
import os
import numpy as np
import warnings
import pickle,re
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

#each batch contains 12 clusters, each with npoints = 200 points sampled by fps

class KittiAdbscanDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root   

        
        self.gt_avail_in_test = args.gt_avail_in_test
        self.split = split
        self.npoints = args.num_point           
        self.catfile = os.path.join(self.root, 'kitti_adbscan_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        shape_ids = {} #empty dictionary with keys 'train'..

        self.train_gt_proposal_file = "list_train.txt" 
        if(split=='train' or (split=='test' and self.gt_avail_in_test==True)):    #when called from train_detection.py: split = test and gt_avail_in_test always True --> list_val.txt
            self.test_proposal_file =  "list_val.txt"                             #when called from test_detection.py: split = test and gt_avail_in_test decides which to use
        elif(split=='test' and self.gt_avail_in_test==False):
            self.test_proposal_file =  "list_test.txt"

        if(split=='train'):                       
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, self.train_gt_proposal_file))]
        
        if(split=='test'):            
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, self.test_proposal_file))] # use validation for test split
        #test proposal file may or may not have gt, train_gt_proposal_file always has
               
        #print(shape_ids)
        #input()
        assert (split == 'train' or split == 'test')
        
        shape_names = []        
        self.fnames = []
        self.gtboxes = []
        self.propboxes = []
        

        if(split=='train' or (split=='test' and self.gt_avail_in_test==True)): 
            for x in shape_ids[split]:   #for either split, read gt and proposal
                z = re.split(r'_|\s',x)                        
                if(z[0]=='Person'):                
                    shape_names.append('Person_sitting')
                    self.fnames.append('_'.join([z[0],z[1],z[2],z[3]]))
                    self.gtboxes.append(np.array([float(z[4]),float(z[5]),float(z[6]),float(z[7]),float(z[8]),float(z[9]),float(z[10])]))
                    self.propboxes.append(np.array([float(z[11]),float(z[12]),float(z[13]),float(z[14]),float(z[15]),float(z[16]),float(z[17])]))                            
                else:
                    shape_names.append(z[0])
                    self.fnames.append('_'.join([z[0],z[1],z[2]]))
                    self.gtboxes.append(np.array([float(z[3]),float(z[4]),float(z[5]),float(z[6]),float(z[7]),float(z[8]),float(z[9])]))
                    self.propboxes.append(np.array([float(z[10]),float(z[11]),float(z[12]),float(z[13]),float(z[14]),float(z[15]),float(z[16])])) 
            
            #gt_box_mid[0], gt_box_mid[1], gt_box_mid[2], gt_velo_dx, gt_velo_dy, gt_velo_dz, gt_velo_rz
            #prop_m[0],prop_m[1],prop_m[2],prop_bbox_shorter,prop_bbox_longer,prop_bbox_velo_dz,prop_shorter_rz 
            #print(shape_names)
            #input()
        
        elif(split=='test' and self.gt_avail_in_test==False):
            
            for x in shape_ids[split]:   #for test split read proposal
                z = re.split(r'\s',x)                    
                shape_names.append('Test') 
                self.fnames.append(z[0])
                self.propboxes.append(np.array([float(z[1]),float(z[2]),float(z[3]),float(z[4]),float(z[5]),float(z[6]),float(z[7])]))         
                #format(fname,prop_m[0],prop_m[1],prop_m[2],prop_bbox_shorter,prop_bbox_longer,prop_bbox_velo_dz,prop_shorter_rz ))    
                
        
        else:
            print("Error!")
            assert(0)
        

        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], self.fnames[i]) ) for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        
        #print(self.datapath)        
        #print(self.gtboxes)
        #print(self.propboxes)
        #input()
                

    def __len__(self):
        return len(self.datapath)
    
    def _get_item(self, index):
        
        fn = self.datapath[index] #includes path
        #print(fn)
          
        point_set = np.load(fn[1]).astype(dtype='float32')  #load default is float64 which causes error in farthest_pt_sample
       
             
        point_set = farthest_point_sample(point_set, self.npoints)   
        #print('after fps')        
        #print(point_set)           
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            
        #print('after pc_norm')  
        #if not self.use_normals:
        #    #point_set = point_set[:, 0:3]
        #print(point_set)

        #input()
        
        propbox = self.propboxes[index]
        fname = self.fnames[index]   #no path

        if(self.split=='train' or (self.split=='test' and self.gt_avail_in_test==True)): 
            cls = self.classes[self.datapath[index][0]] #class index from dict
            label = np.array([cls]).astype(np.int32)
            gtbox = self.gtboxes[index]             
            return point_set, label[0], gtbox, propbox, fname #label is np array with 1 element

        elif(self.split=='test' and self.gt_avail_in_test==False):
            return point_set, propbox, fname #label is np array with 1 element
     
        #else:
        #    print("Error!")
        #    assert(0)
        


    def __getitem__(self, index):
        return self._get_item(index)


#for direct call below 
def parse_args():
  
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=9, type=int,  help='training on Kitti_Adbscan')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=200, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')    
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--data_path', type=str,  help='data path')
    parser.add_argument('--data_path', type=str,  help='data path')                                                                      
    parser.add_argument('--gt_avail_in_test', action='store_true', default = True,  help='gt available in test proposal file')      
    return parser.parse_args()


if __name__ == '__main__':
   

    args = parse_args()
   
    data = KittiAdbscanDataLoader(root=args.data_path, args=args, split='train',caller='train')
    DataL = DataLoader(data, batch_size=2, shuffle=True)
    for point, label, gtbox, propbox in DataL:   
        print(point)
        print(label)
        print(gtbox)
        print(propbox)
    
        input()
