import numpy as np
import glob
import cv2
import os
from tqdm import tqdm
from utils import utils as u
import json
from utils import s3d_utils as su
from scipy.ndimage import gaussian_filter
from collections import defaultdict

datasetlistpt='E:/FCR/PGNET/datasets/raw/datasets.txt'
floorsp_ano_dir='E:/FCR/PGNET/datasets/raw/relabel'
floorsp_hm_dir='E:/FCR/Dataset/Floorsp'
floorsp_depth_dir='E:/FCR/PGNET/datasets/raw/256'
floorsp_input_dir='E:/FCR/Dataset/PGNET/data_check'

"处理为heat所需数据集"
heat_dst_anno_dir='E:/FCR/heat-master/heat-master/data/floorsp'

ano_dir='F:/git/PGNET/Structure3D/annot'
den_dir='E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan/density'
dst_dir='E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan/annotim'
hm_dir='E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan/heatmap'
nor_dir='E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan/normals'
dataset_dir='E:/FCR/Dataset/heat_datasets'
dir_ori='F:/git/PGNET/Structure3D/montefloor_data_all/montefloor_data/'
dst_dir_ori='F:/git/PGNET/Structure3D/heat_datasets_2/'


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

def blur(img,sigma):
    img=gaussian_filter(img,sigma)
    maxi = np.amax(img)
    img = img * 255/maxi
    return img

# u.makedir(hm_dir)
def get_polygons(annos):
    polygons=su.parse_floor_plan_polys(annos)
    floor_map, polygons_list=su.generate_floorplan(annos,polygons,256,256,ignore_types=['door','window','outwall'])
    return floor_map

#从p1-p2线段中获取离散线段点
def get_plist(p1,p2,inter):
    x1,y1=p1
    x2,y2=p2
    # print(x1,x2,y1,y2)
    dis=((x1-x2)**2+(y1-y2)**2)**0.5
    if dis>=12 and dis<16:
        n=3
    else:
        n=max(int((dis/inter)+1),2) 
    xlist=np.linspace(start = x1, stop =x2, num = n, dtype = int)
    ylist=np.linspace(start = y1, stop =y2, num = n, dtype = int)
    plist=[(xlist[i],ylist[i]) for i in range(n)]
    return plist

#生成heat能处理的数据集
def get_anno(lines):
    annot=defaultdict(list)
    for l in lines:
        p1,p2=map(lambda t:tuple(t),(l[0],l[1]))
        annot[p1].append(p2)
        annot[p2].append(p1)
    return annot
#从floorsp数据集中获取连线数组
#lines: [[(x1,y1),(x2,y2)],....]
def floorsp_get_lines(annos):
    lines_dic=annos['seg'][0]['lines']
    lines=[[(l['line_x'][0],l['line_y'][0]),(l['line_x'][1],l['line_y'][1])] for l in lines_dic]
    # lines=annos['edges']
    return lines

#基于corners,edges获取mask
def render(edges,render_pad=0, edge_linewidth=6, corner_size=3, scale=1.):
    size = int(256 * scale)
    mask = np.ones((2, size, size)) * render_pad

    # corners = np.round(corners.copy() * scale).astype(np.int)
    for e in edges:
        mask[0] = cv2.line(mask[0], e[0],e[1], 1.0, thickness=edge_linewidth)
        for c in e:
            mask[1] = cv2.circle(mask[1], c, corner_size, 1.0, -1)

    return mask

class FloorPlan(object):
    def __init__(self):
        pass
    
    def generate_heatmap_structure3D(self):
        wrong={'train':[],'test':[],'valid':[]}
        for phrase in ['train','test','valid']:
            dir=dir_ori+phrase
            dst_dir=dst_dir_ori+phrase
            for f in tqdm(glob.glob(dir+'/*')):
                name=f.split('/')[-1].split('\\')[-1].split('_')[-1].split('.')[0]
                try:
                    ac = np.load(f+'/annot.npy', allow_pickle=True)
                except:
                    wrong[phrase].append(name)
                    continue
                with open(f+'/annotation_3d.json','r') as file:
                    str = file.read()
                    annos = json.loads(str)
                    room_map=get_polygons(annos)
                    
                # boundmask=np.zeros((256,256,3))
                # heatmap=np.zeros((256,256,3))
                ac=ac.item()
                points_map=np.zeros((256,256))
                vertexs_map=np.zeros((256,256))
                
                for (k,v) in ac.items():
                    k,v=map(lambda t:np.array(t).astype(int),(k,v))
                    cv2.circle(vertexs_map,tuple(k),2,(255,255,255),-1)
                    for p in v:
                        plist=get_plist(k,p,12)
                        
                        for ps in plist:
                            cv2.circle(points_map,tuple(ps),2,(255,255,255),-1)
                        cv2.line(room_map,tuple(k),tuple(p),(0,0,0),5)
                        # cv2.line(room_map,tuple(k),tuple(p),(0,0,0),5)
                room_map = cv2.erode(room_map.copy(), kernel, 10)
                room_map = cv2.dilate(room_map.copy(), kernel, 10)
                # print(os.path.join(dst_dir,name+'.png'))
                sample=cv2.imread(os.path.join(dst_dir,name+'.png'))
                newsample=np.zeros((256*4,256,3))
                try:
                    newsample[:256,:,:]=sample[:256,:,:]
                except:
                    wrong[phrase].append(name)
                    continue
                newsample[-256:,:,:]=sample[-256:,:,:]
                newsample[256:2*256,:,:]=room_map*255
                newsample[2*256:3*256,:,0]=np.array(1.3*blur(points_map,2)).clip(0,255)
                newsample[2*256:3*256,:,1]=np.array(1.3*blur(vertexs_map,2)).clip(0,255)
                
                cv2.imwrite(os.path.join(dst_dir,name+'.png'),newsample)
        print(wrong)
                
    def generate_heatmap_floorsp(self):
        with open(datasetlistpt,'r') as f:
            lines =f.readlines()
            for l in tqdm(lines):
                phrase=l.split(':')[0]  
                dst_dir=os.path.join(floorsp_hm_dir,phrase)     
                name=l.split(':')[1].split('\n')[0]
                
                normal_pt=os.path.join(floorsp_input_dir,name+'_normal.png')
                density_pt=os.path.join(floorsp_input_dir,name+'_density.png')
                depth_pt=os.path.join(floorsp_depth_dir,name+'/depth.png')
                ano_pt=os.path.join(floorsp_depth_dir,name+'/{}.json'.format(name))
                heat_annot_pt=os.path.join(heat_dst_anno_dir,'annot/'+name+'.npy')
                txt_pt=os.path.join(heat_dst_anno_dir,phrase+'_list.txt')
                
                with open(ano_pt,'r') as file:
                    str = file.read()
                    annos = json.loads(str)
                    lines=floorsp_get_lines(annos)
                    annos=get_anno(lines)
                    np.save(heat_annot_pt, annos)
                with open(txt_pt, 'a', encoding="utf-8") as f:
                    f.write(name+'\n')
                gt_mask = render(lines)[0]
                gt_mask = 1 - gt_mask
                gt_mask = gt_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(gt_mask, connectivity=4)
                
                background_label = region_mask[0, 0]
                background=region_mask==background_label
                region_mask=(1-background)*region_mask
                
                points_map,vertexs_map,line_map=np.zeros((3,256,256))
                sample=np.zeros((256*4,256,3))
                for l in lines:
                    k,p=l
                    cv2.circle(vertexs_map,tuple(k),2,1.,-1)
                    cv2.circle(vertexs_map,tuple(p),2,1.,-1)
                    plist=get_plist(k,p,8)
                            
                    for ps in plist:
                        cv2.circle(points_map,tuple(ps),2,1.,-1)
                    can=np.zeros((256,256))
                    cv2.line(can,tuple(k),tuple(p),1.,3)
                    can=blur(can,2)
                    line_map=np.where(can>line_map,can,line_map)
                
                normal=cv2.imread(normal_pt)
                density=cv2.imread(depth_pt)
                sample[:256,:,:]=normal
                sample[:256,:,2]=density[:,:,0]
                sample[256:256*2,:,0]=region_mask*255
                sample[256*2:256*3,:,0]=np.array(1.3*blur(points_map,2)).clip(0,255)
                sample[256*2:256*3,:,1]=np.array(1.3*blur(vertexs_map,2)).clip(0,255)
                sample[-256:,:,0]=line_map
                
                cv2.imwrite(os.path.join(dst_dir,name+'.png'),sample)
                
    def split_datasets(self,phrase):
        dst_dir=os.path.join(dataset_dir,phrase)
        u.makedir(dst_dir)
        txtpt=os.path.join('E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan','{}_list.txt'.format(phrase))
        with open(txtpt,'r') as f:
            lines =f.readlines()
            for l in tqdm(lines):
                l=l.split('\n')[0]+'.png'
                density=cv2.imread(os.path.join(den_dir,l))
                normal=cv2.imread(os.path.join(nor_dir,l))
                input=np.maximum(density, normal)
                output=cv2.imread(os.path.join(hm_dir,l))
                im=np.zeros((512,256,3))
                im[:256,:,:]=input
                im[256:,:,:]=output
                cv2.imwrite(os.path.join(dst_dir,l),im)
                
f=FloorPlan()
f.generate_heatmap_floorsp()
# f.split_datasets('train')
# f.split_datasets('test')
# f.split_datasets('valid')