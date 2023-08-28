"""
Author: Atul Divekar
Date: June 2023
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.KittiAdbscanDataLoader import KittiAdbscanDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import box_reg_utils
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=9, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    #parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting')
    parser.add_argument('--data_path', type=str,  help='data path')        
    parser.add_argument('--test_proposal_file', type=str,  help='test proposal file is has gt use: --gt_avail_in_test')        
    parser.add_argument('--gt_avail_in_test', action='store_true', default = False, help='gt available in test proposal file')  
    #flag set to false if not used on command line
    return parser.parse_args()

#can use a proposal file with gt or without

def write_to_file(result_file,fnames,gtboxes,gtbox_preds,IoU3Ds,log_probs,pred_choices,class_targets,orien_log_probs,orien_choices,orien_targets):
       
    #print(result_file)

    #print(gtboxes)
    #print(gtbox_preds)    
    #print(IoU3Ds)
    #print(log_probs)
    #print(pred_choices)
    #print(class_targets)
    #print(orien_log_probs)
    #print(orien_choices)
    #print(orien_targets)

    B = gtboxes.shape[0]
    
    with open(result_file,"a") as f_res:
        for ind in range(B):
            f = fnames[ind]
            gt = gtboxes[ind]
            gtp = gtbox_preds[ind]
            IoU3D = IoU3Ds[ind].item()
            cl_prob = np.exp(log_probs[ind].item())
            cl_tgt = class_targets[ind].item()
            cl_pred = pred_choices[ind].item()
            or_prob = np.exp(orien_log_probs[ind].item())
            or_pred = orien_choices[ind].item()
            or_tgt = orien_targets[ind].item()

            f_res.write("{0} cl_tgt {1} cl_pred {2} gt {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f} {8:.6f} {9:.6f} gtp {10:.6f} {11:.6f} {12:.6f} {13:.6f} {14:.6f} {15:.6f} {16:.6f} IoU3D {17:.6f} cl_prob {18:.6f} or_tgt {19} or_pred {20} or_prob {21:.6f}\n".\
                        format(f, int(cl_tgt), int(cl_pred), gt[0],gt[1],gt[2],gt[3],gt[4],gt[5],gt[6], gtp[0], gtp[1],gtp[2],gtp[3],gtp[4],gtp[5],gtp[6], IoU3D, cl_prob, int(or_tgt), int(or_pred), or_prob))
    

def write_preds_to_file(result_file,fnames,gtbox_preds,log_probs,pred_choices,orien_log_probs,orien_choices):
          
    B = gtbox_preds.shape[0]
    
    with open(result_file,"a") as f_res:
        for ind in range(B):
            f = fnames[ind]            
            gtp = gtbox_preds[ind]            
            cl_prob = np.exp(log_probs[ind].item())           
            cl_pred = pred_choices[ind].item()
            or_prob = np.exp(orien_log_probs[ind].item())
            or_pred = orien_choices[ind].item()
            
            f_res.write("{0} cl_pred {1} gtp {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f} {8:.6f} cl_prob {9:.6f} or_pred {10} or_prob {11:.6f}\n".\
                        format(f, int(cl_pred), gtp[0], gtp[1],gtp[2],gtp[3],gtp[4],gtp[5],gtp[6],  cl_prob, int(or_pred), or_prob))
    

def test(model, loader, result_file, num_class=40, vote_num=1):
   
    classifier = model.eval()
             
    category_instance_acc = np.zeros((num_class, 3))
    orientation_correct_count = 0
    IoU3D_sum = 0
    
    for j, (points, class_target, gtbox, propbox, fname) in tqdm(enumerate(loader), total=len(loader)): #all in cpu
        if not args.use_cpu:
            points, class_target = points.cuda(), class_target.cuda()

        points = points.transpose(2, 1)
        
        #vote_pool = torch.zeros(class_target.size()[0], num_class).cuda()        
        #for _ in range(vote_num):
        #    class_pred, reg_bbox_pred, orien_pred, _ = classifier(points)
        #    class_vote_pool += class_pred        
        #class_pred = vote_pool / vote_num
        
        class_pred, reg_bbox_pred, orien_pred, _ = classifier(points) #points B, 3, N tensor
        log_prob, pred_choice = class_pred.data.max(1)

        #print(points)

        #print('log_prob')
        #print(log_prob)

        #print('pred_choice')
        #print(pred_choice)
        
        #print('class_target')
        #print(class_target)

        gtbox_pred = box_reg_utils.estimate_gt_box(reg_bbox_pred,orien_pred,propbox.cuda())  #all in gpu return gpu
       
        #if gtbox is invalid (all zeros) IoU3D is invalid
        IoU3D = box_reg_utils.IoU_3D(gtbox.cuda(),gtbox_pred)  #all in gpu return gpu                     
        IoU3D_sum +=  IoU3D.cpu().sum()
                
        orien_target = propbox[:,11].cpu() #ind_min from propbox when gt is known 
        orien_log_prob,orien_choice = orien_pred.data.max(1)
        orien_choice = orien_choice.cpu()
        
        #print('gtbox')
        #print(gtbox) 

        #print('gtbox_pred')
        #print(gtbox_pred)
               
        write_preds_to_file(result_file,fname,gtbox_pred,log_prob,pred_choice,orien_log_prob,orien_choice)
        
        for cat in np.unique(class_target.cpu()):               
            category_instance_acc[cat, 1] += (class_target == cat).cpu().sum()  # num instances of cat in class_target
            category_instance_acc[cat, 0] += pred_choice[class_target == cat].eq(cat).cpu().sum()    # of these, correctly predicted
        
        orientation_correct_count += orien_choice.eq(orien_target.long().data).cpu().sum()
    

    category_instance_acc[:, 2] = category_instance_acc[:, 0] / category_instance_acc[:, 1]

    print('category_instance_acc')
    print(category_instance_acc)
   
    totals = np.sum(category_instance_acc[:,0:2],axis=0)
    print('correct classified {0} of {1}'.format(totals[0],totals[1]))
    
    instance_acc = totals[0] / totals[1]
    orien_instance_acc = orientation_correct_count / totals[1]    
    IoU3D_mean = IoU3D_sum / totals[1] 
   
    return instance_acc, IoU3D_mean, orien_instance_acc 

           
def test_only_pred(model, loader, result_file, num_class=40, vote_num=1):
  
    classifier = model.eval()

    for j, (points, propbox, fname) in tqdm(enumerate(loader), total=len(loader)): #all in cpu
        if not args.use_cpu:
            points = points.cuda()

        points = points.transpose(2, 1)
                            
        class_pred, reg_bbox_pred, orien_pred, _ = classifier(points) #points B, 3, N tensor
        log_prob, pred_choice = class_pred.data.max(1)

        #print(points)

        #print('log_prob')
        #print(log_prob)

        #print('pred_choice')
        #print(pred_choice)
                   
        gtbox_pred = box_reg_utils.estimate_gt_box(reg_bbox_pred,orien_pred,propbox.cuda())  #all in gpu return gpu
                  
        orien_log_prob,orien_choice = orien_pred.data.max(1)
        orien_choice = orien_choice.cpu()
          
        #print('gtbox_pred')
        #print(gtbox_pred)
        
        write_preds_to_file(result_file,fname,gtbox_pred,log_prob,pred_choice,orien_log_prob,orien_choice)



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path

    test_dataset = KittiAdbscanDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
   
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    data_src = re.split(r'/',data_path)[-1]  

    result_file = os.path.join(experiment_dir,("result_" + data_src + "_" + args.test_proposal_file) )  
    if os.path.exists(result_file):
        os.remove(result_file)
    print(result_file)
   
    with torch.no_grad():
         
        if(args.gt_avail_in_test==True):  
            instance_acc, IoU3D_mean, orien_instance_acc = test(classifier.eval(), testDataLoader, result_file,  vote_num=1, num_class=num_class)
            log_string('Test Instance Accuracy: %f, IoU3D_mean %f, Orien Instance Accuracy %f' % (instance_acc, IoU3D_mean,orien_instance_acc))
        else:
            test_only_pred(classifier.eval(), testDataLoader, result_file,  vote_num=1, num_class=num_class)


if __name__ == '__main__':
    args = parse_args()
    main(args)

