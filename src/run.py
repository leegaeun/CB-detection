import argparse
import os
from glob import glob
import numpy as np
import cv2

import tensorflow as tf
from DataSet_infer import ModelConfig
from mrcnn import model as modellib


parser = argparse.ArgumentParser(description='')

parser.add_argument('--path_img', dest='path_img', default= '/data', help='Dataset-path containing png')
parser.add_argument('--weight_path', dest='weight_path', default= './weights/model.seg.cb.hdf5', help='Weight-path containing hdf5')
parser.add_argument('--path_dst', dest='path_dst', default= './results', help='Path where output masks will be stored')

args = parser.parse_args()

def getMostProperMask(result, Classes):
    '''
        bboxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        Classes: [num_classes]
    '''
    
    scores = np.asarray(result['scores'])
    masks =  np.asarray(result['masks'])
    bboxes = np.asarray(result['rois'])
    class_ids = np.asarray(result['class_ids'])
    
    refined_scores = []
    refined_masks = []
    refined_bboxes = []
    refined_class_ids = []
    
    # only one that best score per class.
    for class_id in range (len(Classes)):
        
        idxs = np.where(class_ids==class_id)[0]
        if(len(idxs)==0):
            continue
            
        scores_cl = sorted(enumerate(scores[idxs]), reverse=True, key=lambda r:r[1])
        
        idx = idxs[scores_cl[0][0]]
        
        refined_scores.append(scores[idx])
        refined_masks.append(masks[:,:,idx])
        refined_bboxes.append(bboxes[idx])
        refined_class_ids.append(class_id)
    
    if len(refined_masks)>0:
        refined_masks = np.transpose(refined_masks,(1,2,0))
    return {'scores':refined_scores, 'masks':refined_masks, 'rois':refined_bboxes, 'class_ids':refined_class_ids}

def save_result_masks(result, class_names, save_filename, maskDir):
    '''
    maskDir: Directory of predicted instance mask.
    '''
    masks = np.array(result['masks'],dtype='uint8')
    class_ids = result['class_ids']
    
    if len(class_ids) == 0:
        return None
    
    for i, class_id in enumerate (class_ids):
        mask = masks[:,:,i]
        class_name = class_names[class_id]
    
        maskDir_clss = os.path.join(maskDir, class_name)
        if not os.path.isdir(maskDir_clss):
            os.mkdir(maskDir_clss)
    
        _, mask = cv2.threshold(np.asarray(mask,dtype='uint8'), 0, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(maskDir_clss,save_filename), mask)

        
# -------------------------------------       
# RUN        

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path_img = args.path_img
weight_path = args.weight_path
path_dst = args.path_dst

modelConfig = ModelConfig()
model_lineseg = modellib.MaskRCNN(mode="inference", model_dir=os.path.dirname(weight_path), config=modelConfig)
model_lineseg.load_weights(weight_path, by_name=True)
Classes = [None]*modelConfig.NUM_CLASSES
for class_info in modelConfig.class_info:
    Classes[class_info['id']] = class_info['name']
    
for path in glob(os.path.join(path_img,'*.png')):
    print("[Processing] ... ",path)
    img = cv2.imread(path, 1)
    pred = model_lineseg.detect([img], verbose=0)[0]
    pred = getMostProperMask(pred, Classes)
    save_result_masks(pred, 
                      Classes, 
                      '{}.png'.format(os.path.basename(path).split('.')[0]), 
                      path_dst)    
print("[Completed] ! ")
    