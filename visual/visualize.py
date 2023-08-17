import numpy
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import glob
import utils.utils as u
from dataprocess.Douglas import Graph_Optimize
from utils.utils import render
from tqdm import tqdm
from optimization import convert_anno

ColorMap = np.array([
        [241,241,241],
        [251,215,139], 
        [255,241,238], 
        [240,141,144],
        [225,239,250], 
        [98,156,204], 
        [164,175,203], 
        [98,120,167], 
        [199,192,163], 
],dtype=np.int64)


def IoU(mask_A,mask_B):
    mask_a=mask_A/max(1,np.max(mask_A))
    mask_b=mask_B/max(1,np.max(mask_B))
    mask=mask_a+mask_b
    IOU=len(np.argwhere(mask>1))/max(1,len(np.argwhere(mask>0)))
    return IOU

class Rooms(object):
    def __init__(self,vertexs,edges,adjs,vmap,name,outputdir,size=[256,256]):
        self.vertexs=np.array(vertexs,np.int32)
        self.edges=edges
        self.adjs=adjs
        self.vmap=vmap
        self.size=size
        # self.labelregions=labelregions
        rect = (0,0,size[1],size[0])
        subdiv = cv2.Subdiv2D(rect)
        for p in self.vertexs:
            subdiv.insert(p)
        trangleList = subdiv.getTriangleList()
        self.trangleList=[]
        for t in  trangleList:
            pt1 = (int(t[0]),int(t[1]))
            pt2 = (int(t[2]),int(t[3]))
            pt3 = (int(t[4]),int(t[5]))
            self.trangleList.append([pt1,pt2,pt3])
        self.cal_regions()
        F_Generate=Floorplan(self.regions,edges,vertexs,name,outputdir=outputdir)
        # F_Label=Floorplan(label_rooms,label_edges,label_vertexs,'label')
            
    #计算region边界
    def cal_regions(self):
        self.tradjs=np.zeros((len(self.trangleList),len(self.trangleList)))
        p_list=dict()
        for es in self.edges:
            for i,tr in enumerate(self.trangleList):
                for e in [[es[0],es[1]],[es[1],es[0]]]:
                    if e[0] in tr and not e[1] in tr:
                        tr_=tr.copy()
                        tr_.remove(e[0])
                        match=tr_
                        match.append(e[1])
                        find=None
                        for j,tr2 in enumerate(self.trangleList):
                            if i!=j and match[0] in tr2 and match[1] in tr2 and match[2] in tr2:
                                find=j
                                break
                        if find!=None:
                            self.trangleList[i]=[match[0],e[0],e[1]]
                            self.trangleList[j]=[match[1],e[0],e[1]]
        for p in self.vertexs:
            p_list['{}_{}'.format(p[0],p[1])]=[]
        for i,t in enumerate(self.trangleList):
            for p in t:
                p_list['{}_{}'.format(int(p[0]),int(p[1]))].append(i)                
        for k,p in enumerate(self.vertexs):
            ners=[tuple(self.vertexs[n[0]]) for n in np.argwhere(self.adjs[k]>0)]
            for i in p_list['{}_{}'.format(p[0],p[1])]:
                for j in p_list['{}_{}'.format(p[0],p[1])]:
                    if i==j or self.tradjs[i][j]:
                        continue
                    for tp in self.trangleList[i]:
                        if tuple(tp)==tuple(self.vertexs[k]):
                            continue
                        if tp in self.trangleList[j] and not tp in ners:
                            self.tradjs[i][j]=self.tradjs[j][i]=1
        self.regions=[]
        self.tradjs_=self.tradjs.copy()
        can=np.zeros((self.size[0],self.size[0],3))
        while(len(np.argwhere(self.tradjs_>0))>0):
            next=np.argwhere(self.tradjs_>0)[0][0]  
            region=self.get_region_from_graph([],next)
            if self.filtregion(region):
                self.regions.append([self.trangleList[id] for id in region])
    
    def get_region_from_graph(self,region,next):
        region.append(next)
        if len(np.argwhere(self.tradjs_[next]>0))==0:
            return 
        for n in np.argwhere(self.tradjs_[next]>0):
            n=n[0]
            self.tradjs_[next][n]=self.tradjs_[n][next]=0
            self.get_region_from_graph(region,n)
        return region
     

    def checkadj(self,p1,p2):
        n1,n2=[self.vmap['{}_{}'.format(p[0],p[1])] for p in [p1,p2]]
        if self.adjs[n1][n2]==0:
            return False
        return True
                
    #分出外界region和内部region
    def filtregion(self,region):
        for tr in region:
            p1,p2,p3=self.trangleList[tr]
            l12=True
            l23=True
            l13=True
            ners=[self.trangleList[n[0]] for n in np.argwhere(self.tradjs[tr]>0)]
            for trn in ners:
                if p1 in trn and p2 in trn:
                    l12=False
                if p2 in trn and p3 in trn:
                    l23=False
                if p1 in trn and p3 in trn:
                    l13=False   
            if l12 and not self.checkadj(p1,p2):
                return False
            if l23 and not self.checkadj(p2,p3):
                return False
            if l13 and not self.checkadj(p1,p3):
                return False
        return True
    
    def transregion(self,region):
        region_can=[]
        for r in region:
            region_can.append(self.draw_region(r))
        return region_can
    
    #画出region
    def draw_region(self,region):
        region=np.array([region],dtype=int) if len(region[0])==2 else np.array(region,dtype=int)
        can=np.zeros((self.size[0],self.size[0]))
        for trp in region:
            can=cv2.fillConvexPoly(can,np.array(trp,dtype=int), (255, 255, 255))
        return can


class Floorplan(object):
    def __init__(self,regions,edges,vertexs,name,size=256,outputdir='./'):
        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(size/dpi,size/dpi)
        fig.set_frameon(False)
        self.ax = fig.add_axes([0,0,1,1])
        self.ax.set_aspect('equal')
        self.ax.set_xlim([0,255])
        self.ax.set_ylim([0,255])
        self.regions=regions
        self.edges=edges
        self.vertexs=vertexs
        self.size=size
        
        self.draw_edges()
        self.draw_vertexs()
        self.draw_regions()

        fig.canvas.print_figure(os.path.join(outputdir,'0'+name+'.png'))
        
    
    def draw_regions(self):
        for i,boxs in enumerate(self.regions):
            cid=i%len(ColorMap)
            #三角形组成的region
            if len(boxs[0])==3:
                for b in boxs:
                    x=[p[0] for p in b]
                    y=[255-p[1] for p in b]
                    self.ax.fill(x,y,c=ColorMap[cid]/255.0,zorder=1)
            #多边形的boundingbox
            else:
                x=[p[0] for p in boxs]
                y=[255-p[1] for p in boxs]
                self.ax.fill(x,y,c=ColorMap[cid]/255.0,zorder=1)
                    
    def draw_edges(self):
        for e in self.edges:
            x=[p[0] for p in e]
            y=[255-p[-1] for p in e]
            self.ax.plot(x,y,color=(0,0,0),linewidth=1,zorder=2)
    
    def draw_vertexs(self):
        x=[p[0] for p in self.vertexs]
        y=[255-p[1] for p in self.vertexs]
        plt.scatter(x, y, color=(1,1,1), marker='o', edgecolors=(0,0,0), s=7,zorder=3) 

def vis_input(seg=256):
    s3d_dir='E:/FCR/heat-master/heat-master/data/s3d_floorplan/density'
    lj_dir='E:/FCR/Dataset/Floorsp/floorsp/density'
    dst_dir=os.path.join('E:/FCR/heat-master/heat-master/Ablation_results','lj_gt')
    u.makedir(dst_dir)
    for im in tqdm(glob.glob(lj_dir+'/*')):
        name=im.split('/')[-1].split('\\')[-1]
        input=255-cv2.imread(im)
        cv2.imwrite(os.path.join(dst_dir,name),input)

def vis_ablation(seg=256):
    # vis_name=['s3d_gt','1','2','3','4','5','pg','gt']
    vis_name=['floorsp','heat','roomformer','polygraph','gt']
    # vis_dir=[os.path.join('E:/FCR/heat-master/heat-master/Ablation_results','{}_vis'.format(id)) for id in vis_name]
    # dst_dir='E:/FCR/heat-master/heat-master/Ablation_results/ablation'
    vis_dir=[os.path.join('E:/FCR/heat-master/heat-master/results','viz_{}_s3d_256'.format(id)) for id in vis_name]
    dst_dir='E:/FCR/heat-master/heat-master/results/s3d'
    u.makedir(dst_dir)
    for data in tqdm(os.listdir(vis_dir[3])):
        ims=[cv2.imread(os.path.join(dir,data)) for dir in vis_dir[1:]]
        newim=np.zeros((5*seg,seg,3))
        newim[:seg,:,:]=cv2.imread(os.path.join(vis_dir[0],str(int(data.split('.png')[0])-3250))+'_recon.png')
        for i,im in enumerate(ims):
            start=seg*(i+1)
            newim[start:start+seg,:,:]=im
        cv2.imwrite(os.path.join(dst_dir,data),newim)

#process different data format
def process_annot(anno,npy_pt):
    new_anno={}
    graph=anno['global_graph']
    edges=[]
    corners=list(graph.keys())
    for k,vs in graph.items():
        for v in vs:
            edges.append((corners.index(k),corners.index(v)))
    new_anno['corners']=np.array(corners,dtype=int)
    new_anno['edges']=np.array(edges,dtype=int)
    np.save(npy_pt,new_anno)
    return new_anno
    
if __name__=='__main__':
    # vis_ablation()
    visual_npy_dir='E:/FCR/heat-master/heat-master/results/npy_roomformer_s3d_256'
    # visual_npy_dir='E:/FCR/floor-sp-master/floor-sp/s3d_results_floorplan/final/round_1/results'
    # visual_npy_dir='E:/FCR/heat-master/heat-master/Ablation_results/4'
    # visual_npy_dir='E:/FCR/Dataset/heat/heat_data/data/s3d_floorplan/annot'
    # visual_npy_dir='E:/FCR/heat-master/heat-master/data/floorsp/annot'
    visual_output_dir='E:/FCR/heat-master/heat-master/results/viz_roomformer_s3d_256'
    # visual_npy_dst_dir='E:/FCR/heat-master/heat-master/results/npy_floorsp_s3d_256'
    u.makedir(visual_output_dir)
    for pt in tqdm(glob.glob(visual_npy_dir+'/*')):
        name=pt.split('//')[-1].split('\\')[-1].split('/')[-1].split('.npy')[0]
        annot = np.load(pt, allow_pickle=True, encoding='latin1').tolist()
        # npy_pt=os.path.join(visual_npy_dst_dir,str(103250+int(name.split('_recon')[0]))[1:]+'.npy')
        # annot=process_annot(annot,npy_pt)
        # continue
        # annot=convert_anno(name)
        corners=np.array(annot['corners'])
                         
        edges=np.array(annot['edges'])

        gt_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
        gt_mask = 1 - gt_mask
        gt_mask = gt_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(gt_mask, connectivity=4)
        background_label = region_mask[0, 0]
        all_gt_masks = []
        all_gt_polys=[]
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            the_poly=u.polygonize_mask(the_region) 
            the_poly=[[p[1],p[0]] for p in the_poly]
            all_gt_polys.append(the_poly) 
            
        def poly_map_sort_key(x):
            return np.sum(x[1])
        gt_polys_sorted_indcs = [i[0] for i in sorted(enumerate(all_gt_polys), key=poly_map_sort_key, reverse=True)]
        all_gt_polys = [all_gt_polys[ind] for ind in gt_polys_sorted_indcs]
        
        edges=[(tuple(corners[e[0]]),tuple(corners[e[1]])) for e in edges]
        vertexs,edges,adj,vmap=u.get_adj_from_edges(edges)
        F_Generate=Floorplan(all_gt_polys,edges,vertexs,name,outputdir=visual_output_dir)
        