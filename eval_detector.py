import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    b1tl_row, b1tl_col, b1br_row, b1br_col = box_1[:]
    
    b2tl_row, b2tl_col, b2br_row, b2br_col = box_2[:]
    
    xleft = max(b1tl_row, b2tl_row)
    ytop = max(b1tl_col, b2tl_col)
    xright = max(b1br_row, b2br_row)
    ybottom = max(b1br_col, b2br_col)
    
    if (xright < xleft) or (ybottom < ytop):
        iou = 0
    else:
        int_area = (xright - xleft)*(ybottom-ytop)
    
        bb1_area = (b1br_row-b1tl_row)*(b1br_col -b1tl_col)
        bb2_area = (b2br_row-b2tl_row)*(b2br_col -b2tl_col)
    
        iou = int_area / float(bb1_area + bb2_area - int_area)
    iou = min(1, iou)
    iou = max(0, iou)
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''

    for pred_file in preds.keys():
        pred = preds[pred_file]
        gt = gts[pred_file]
        for i in range(len(gt)):
            detects = []
            #prd = len(pred[:][:4])
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if (iou > iou_thr):
                    detects.append(iou)
            if len(detects) > 1:
                FP += 1
                TP += 1
            elif len(detects) == 1:
                TP += 1
            else:
                FN += 1
                


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/hw02_preds' #'../data/hw01_preds'
gts_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/hw02_annotations'

# load splits:
split_path = 'C:/Users/kbdra/OneDrive/Documents/Caltech (classes)/Senior year/Spring 2021/Computer Vision/hw2_data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 


#confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds

confidence_thrs = []
for fname in preds_train:
    for pred in preds_train[fname]:
        confidence_thrs.append(pred[4])
confidence_thrs = np.sort(np.array(confidence_thrs))

tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves
precision = np.divide(tp_train, fp_train)
recall = np.divide(tp_train, sum(fn_train, tp_train))
plt.plot(recall, precision)

if done_tweaking:
    print('Code for plotting test set PR curves.')
