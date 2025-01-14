"""
Author: Atul Divekar
Date: June 2023
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.KittiAdbscanDataLoader import KittiAdbscanDataLoader

import box_reg_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg_adbscan', help='model name [default: pointnet2_cls_ssg_adbscan]')
    parser.add_argument('--num_category', default=9, type=int,  help='training on Kitti_Adbscan')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=200, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--data_path', type=str,  help='data path')   
    parser.add_argument('--train_gt_proposal_file', type=str,  help='train gt proposal file') 
    parser.add_argument('--test_proposal_file', type=str,  help='test proposal file with gt') 
    parser.add_argument('--gt_avail_in_test', action='store_true', default = True,  help='gt available in test proposal file')  
    #gt always present in test proposal file, above line sets the flag to True even when not used on cmd line as needed
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, num_class=40):
    
    category_instance_acc = np.zeros((num_class, 3))
    orientation_correct_count = 0
    IoU3D_sum = 0

    classifier = model.eval()

    for j, (points, class_target, gtbox, propbox, fname) in tqdm(enumerate(loader), total=len(loader)):
            
        if not args.use_cpu:
            points, class_target = points.cuda(), class_target.cuda()  

        points = points.transpose(2, 1)
        class_pred, reg_bbox_pred, orien_pred, _ = classifier(points)
        pred_choice = class_pred.data.max(1)[1]
        
        #print(points)

        #print('pred_choice')
        #print(pred_choice)
        #print('class_target')
        #print(class_target)

        gtbox_pred = box_reg_utils.estimate_gt_box(reg_bbox_pred,orien_pred,propbox.cuda())  #all tensors in gpu returns in gpu
        
        IoU3D = box_reg_utils.IoU_3D(gtbox.cuda(),gtbox_pred)
        IoU3D_sum +=  IoU3D.cpu().sum()

        orien_target = propbox[:,11].cpu()
        orien_choice = orien_pred.data.max(1)[1].cpu()

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
    IoU_mean = IoU3D_sum / totals[1] 
              
    return instance_acc, IoU_mean, orien_instance_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path  

    #for train: for each cluster load gnd truth box
   
    train_dataset = KittiAdbscanDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = KittiAdbscanDataLoader(root=data_path,  args=args, split='test', process_data=args.process_data)
        
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0   
    best_IoU_mean = 0.0


    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        category_instance_acc = np.zeros((num_class, 3))
        orientation_correct_count = 0
        IoU3D_sum = 0

        classifier = classifier.train()
       
        scheduler.step()
               
        for batch_id, (points, class_target, gtbox, propbox, fname) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            
            points[:, :, 0:3], gtbox, propbox = provider.random_scale_point_cloud_and_box(points[:, :, 0:3],gtbox,propbox) #scale, shift gt box accordingly, generate propbox 
            points[:, :, 0:3], gtbox, propbox = provider.shift_point_cloud_and_box(points[:, :, 0:3],gtbox,propbox)
           
            '''
            #gtbox, propbox Tensors kept in cpu 
            nanP = np.any(np.isnan(points))
            if(nanP):
                print("nan present2")
            else:
                print("no nan")

            infP = np.any(np.isinf(points))
            if(infP):
                print("inf present2")
            else:
                print("no inf")
            '''

            points = torch.Tensor(points)
            points = points.transpose(2, 1) #B,N,3 -> B,3,N           
            if not args.use_cpu:
                points, class_target = points.cuda(), class_target.cuda()   #gtbox propbox in cpu
            

            '''
            is_param_nan = torch.stack([torch.isnan(p).any() for p in classifier.parameters()]).any()
            is_param_inf = torch.stack([torch.isinf(p).any() for p in classifier.parameters()]).any()

            
            if(is_param_nan):
                print("param nan present")
            else:
                print("no param nan")

            if(is_param_inf):
                print("param inf present")
            else:
                print("no param inf")
            '''

            class_pred, reg_bbox_pred, orien_pred, trans_feat = classifier(points)            
            #class_target, class_pred, reg_bbox_pred, orien_pred are Tensors in GPU
            
            #class_pred:  batch_sz * categories tensor -- for each member in batch log(prob) of being in each cat.
            pred_choice = class_pred.data.max(1)[1] #for each row, find max, get index (from [1])
            
            orien_target = propbox[:,11].cpu()
            orien_choice = orien_pred.data.max(1)[1].cpu()
            

            '''
            print('\nnew batch ground truth')
            print('class_target')
            print(class_target)
            print('gtbox')
            print(gtbox)
            print('propbox')    
            print(propbox)    

            print('\npreds for new batch')
            print('class_pred')
            print(class_pred)           
            print('pred_choice')
            print(pred_choice) 
            print('reg_bbox_pred')
            print(reg_bbox_pred)
            print('orien_pred')
            print(orien_pred)
            
            print('orien_choice')
            print(orien_choice)
            '''
           
            

            loss = criterion(class_pred, class_target.long(), reg_bbox_pred, propbox, gtbox, orien_pred) #Tensors propbox, gtbox on cpu class_target on gpu                   
            gtbox_pred = box_reg_utils.estimate_gt_box(reg_bbox_pred,orien_pred,propbox.cuda())  #all inputs are Tensors in gpu, returns in gpu

            IoU3D = box_reg_utils.IoU_3D(gtbox.cuda(),gtbox_pred)   
            IoU3D_sum +=  IoU3D.cpu().sum()
            #print('IoU')  
            #print(IoU)     
            #input()

            for cat in np.unique(class_target.cpu()):              
                category_instance_acc[cat, 1] += (class_target == cat).cpu().sum()  # num instances of cat in class_target
                category_instance_acc[cat, 0] += pred_choice[class_target == cat].eq(cat).cpu().sum()    # of these, correctly predicted
        
            orientation_correct_count += orien_choice.eq(orien_target.long().data).cpu().sum()
    
            loss.backward()
            
            '''
            is_param_nan = torch.stack([torch.isnan(p).any() for p in classifier.parameters()]).any()

            if(is_param_nan):
                print("param nan present2")
            else:
                print("no param nan2")

            is_param_inf = torch.stack([torch.isinf(p).any() for p in classifier.parameters()]).any()
            if(is_param_inf):
                print("param inf present2")
            else:
                print("no param inf2")
            '''

            for name, param in classifier.named_parameters():
                if torch.isnan(param.grad).any():
                    print("{} nan gradient found".format(name))
                if torch.isinf(param.grad).any():
                    print("{} inf gradient found".format(name))




            optimizer.step()

            is_param_nan = torch.stack([torch.isnan(p).any() for p in classifier.parameters()]).any()

            '''
            if(is_param_nan):
                print("param nan present3")
            else:
                print("no param nan3")
            '''

            global_step += 1

        category_instance_acc[:, 2] = category_instance_acc[:, 0] / category_instance_acc[:, 1]

        print('category_instance_acc')
        print(category_instance_acc)
   
        totals = np.sum(category_instance_acc[:,0:2],axis=0)
        print('correct classified {0} of {1}'.format(totals[0],totals[1]))
    
        train_instance_acc = totals[0] / totals[1]
        orien_train_instance_acc = orientation_correct_count / totals[1]    
        IoU_train_mean = IoU3D_sum / totals[1]  

        log_string('Train Instance Accuracy: %f' % train_instance_acc)               
        log_string('Train Orien Instance Accuracy: %f' % orien_train_instance_acc)       
        log_string('Train IoU Mean: %f' % IoU_train_mean)
        
        
        with torch.no_grad():
            instance_acc,  IoU_mean, orien_instance_acc  = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (IoU_mean >= best_IoU_mean):
                best_IoU_mean = IoU_mean 
                best_epoch = epoch + 1   
            
            log_string('Test Instance Accuracy: %f' % (instance_acc))
            log_string('Best Instance Accuracy: %f' % (best_instance_acc))
            log_string('Test IoU Mean: %f' % (IoU_mean))
            log_string('Test Orien Instance Accuracy: %f' % (orien_instance_acc))

            if (instance_acc >= best_instance_acc or IoU_mean >= best_IoU_mean):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,                    
                    'IoU_mean': IoU_mean,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            
            #input()

        global_epoch += 1
        
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()    
    torch.set_printoptions(sci_mode=False,linewidth=150)
    main(args)
