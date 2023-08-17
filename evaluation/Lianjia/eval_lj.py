import numpy as np
import os
from Lianjia_metric import compute_metrics,get_recall_and_precision
from planar_graph_utils import get_cut_point
from tqdm import tqdm
import cv2
import os
import glob

anno_dir='./annot'
npy_dir='../npyoutput/Lianjia'


def convert_anno(name):
    # anno_path=os.path.join(anno_dir,name.split('.png')[0].split('.jpg')[0]+'.npy')
    anno_path=os.path.join(anno_dir,name)
    annot = np.load(anno_path, allow_pickle=True, encoding='latin1').tolist()
    gt_data=dict()
    gt_data_corners=list(annot.keys())
    gt_data['corners']=np.array(list(gt_data_corners))
    edge=[]
    for p in gt_data_corners:
        pv=annot[p]
        for v in pv:
            if not (p,v) in edge and  not (v,p) in edge:
                edge.append((p,v))
    edge=[(gt_data_corners.index(e[0]),gt_data_corners.index(e[1])) for e in edge]
    gt_data['edges']=np.array(edge)
    return gt_data


def cal_metric(gt_datas,pred_datas):

    corner_tp = 0.0
    corner_fp = 0.0
    corner_length = 0.0

    region_tp = 0.0
    region_fp = 0.0
    region_length = 0.0
    
    angle_tp=0.0
    angle_gt_length=0.0
    angle_pred_length=0.0
    per_angle=0.0
    
    cut_point=0.0
    room_overlap=0.0
    for gt_data,pred_data in zip(gt_datas,pred_datas):
        score=compute_metrics(gt_data, pred_data)
        
        cut_point+=get_cut_point(pred_data)
        room_overlap+=score['room_overlap']
        
        corner_tp += score['corner_tp']
        corner_fp += score['corner_fp']
        corner_length += score['corner_length']
        
        region_tp += score['region_tp']
        region_fp += score['region_fp']
        region_length += score['region_length']
        angle_tp+=score['angle_tp']
        per_angle+=score['per_angle']
        
        angle_gt_length+=score['gt_angle_length']
        angle_pred_length+=score['pred_angle_length']
        

    recall, precision = get_recall_and_precision(corner_tp, corner_fp, corner_length)
    f_score = 2.0 * precision * recall / (recall + precision + 1e-8)
    print('corners - precision: %.3f recall: %.3f' % (precision, recall))

    ## region
    recall, precision = get_recall_and_precision(region_tp, region_fp, region_length)
    print('regions - precision: %.3f recall: %.3f ' % (precision, recall))
    
    recall=angle_tp/(angle_gt_length + 1e-8)
    precision=angle_tp/(angle_pred_length + 1e-8)
    per_angle+=(angle_gt_length-angle_tp)*10
    print('angles - precision: %.3f recall: %.3f' % (precision, recall))
    
    structral=10*(cut_point+room_overlap+1)/(1+corner_tp+region_tp)
    print('structral - S_cons: %.3f  MAnE: %.3f S_comp: %.3f ' % (structral,per_angle/(angle_gt_length + 1e-8),region_tp/ (corner_tp+corner_fp + 1e-8)))


if __name__=="__main__":
    gt_datas=[]
    pred_datas=[]
    for nypt in tqdm(glob.glob(npy_dir+'/*')):
        # img=cv2.resize(cv2.imread(impt),(segsize,3*segsize))
        name=nypt.split('//')[-1].split('\\')[-1].split('/')[-1]
        gt_data=convert_anno(name)
        pred_data=np.load(nypt,allow_pickle=True).tolist()
        gt_datas.append(gt_data)
        pred_datas.append(pred_data)
    cal_metric(gt_datas,pred_datas)