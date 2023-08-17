import argparse
import numpy as np
import scipy.ndimage.filters as filters
import cv2
import glob
import os
from tqdm import tqdm
import utils.utils as u
import math
from dataprocess.Douglas import Graph_Optimize

from configs.Graphoptimization import *


experiment='A'
dataname='Lianjia'
# dataname='Structure3d'

#dir of heatmaps 
heatmap_dir='./experiment/{}/output'.format(experiment)

#dir to save process imgs
process_dir='./experiment/{}/process'.format(experiment)
#dir to save npy files
npy_dir='./evaluation/npyoutput/{}'.format(dataname)
u.makedir(npy_dir)

#Merge points on a split block
def fitdot(segcan):
    # print(segcan)
    points=np.argwhere(segcan>0)
    res=np.zeros(segcan.shape)
    num=len(points)
    x=0
    y=0
    if num>0:
        for p in points:
            x+=p[0]
            y+=p[1]
        x=int(x/num)
        y=int(y/num)
        res[x][y]=255
        # cv2.circle(res,(y,x),1,(255,255,255),-1)
    return res        


#Merge points in a block       
def fitbasedot(vertex,seg):
    points=np.argwhere(vertex>255*point_threshold)
    
    w,h=vertex.shape
    # vertex=np.zeros((w,h,3))
    for p in points:
        segcan=np.zeros((2*seg,2*seg))
        l=max(0,p[0]-seg)
        r=min(segsize,p[0]+seg)
        d=max(0,p[1]-seg)
        u=min(segsize,p[1]+seg)
        segcan[:r-l,:u-d]=vertex[l:r,d:u].copy()
        fit=fitdot(segcan)
        vertex[l:r,d:u]=fit[:r-l,:u-d]
        # cv2.circle(vertex,(p[1],p[0]),4,(0,0,255),0)
    return vertex

#apply NMS operation removes redundant points
def corner_nms(data):
    neighborhood_size = 5
    threshold = 0

    # for i in range(len(preds)):
    #     data[preds[i, 1], preds[i, 0]] = confs[i]

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    maxima[data<255*point_threshold]=0

    results = np.where(maxima > 0)
    filtered_preds = np.stack([results[1], results[0]], axis=-1)

    new_confs = list()
    for i, pred in enumerate(filtered_preds):
        new_confs.append(data[pred[1], pred[0]])
    new_confs = np.array(new_confs)/255

    return filtered_preds, new_confs

#gets the vector point set from the heatmap 
def get_corners(img):
    points_data=img[-segsize:,:,0]
    # vertexs_data=img[-segsize:,:,1]
    points,points_conf=corner_nms(points_data)
    # vertexs,vertexs_conf=corner_nms(vertexs_data)
    can=np.zeros((segsize,segsize))
    for i,p in enumerate(points):
        color=int(points_conf[i]*255)
        can[p[1]][p[0]]=color
        # cv2.circle(can,tuple(p),2,(color,color,color),-1)
    can_=fitbasedot(can.copy(),3)
    # can_=can
    return can_


def rect_contains(rect,point):
    if point[0] <rect[0]:
        return False
    elif point[1]<rect[1]:
        return  False
    elif point[0]>rect[2]:
        return False
    elif point[1] >rect[3]:
        return False
    return True


# optimization based on a sample, a graph structure
class Graph(object):
    def __init__(self,img,name,iflabel=False):
        self.name=name
        can=get_corners(img)
        points=np.argwhere(can>0)
        self.vertexs=[(int(p[1]),int(p[0])) for p in points] if not iflabel else [(int(p[0]),int(p[1])) for p in points]
        self.ori_vertexs=self.vertexs
        
        # self.edgeconf=1-u.data_augment(img[segsize:segsize*2,:,0],3)/255
        self.edgeconf=1-img[:segsize,:,0]/255
        self.regionconf=img[segsize:segsize*2,:,0]/255
        self.size = img.shape[1]
        self.rect = (0,0,self.size,self.size)

    #Triangulation based on a set of vertices
    def draw_delaunay(self):
        self.subdiv = cv2.Subdiv2D(self.rect)

        self.vnum=len(self.vertexs)
        # self.vmap=dict()
        for i,p in enumerate(self.vertexs):
            self.subdiv.insert(p) # Insert a single point into a Delaunay triangulation.
        trangleList = self.subdiv.getTriangleList()

        # r = (0,0,self.size,self.size)
        delps=[]
        # pts=[]
        # edges_of_regularTri = []
        self.adjs=np.zeros((self.vnum,self.vnum))
        self.trangleList=[]
        for t in  trangleList:
            pt1 = (int(t[0]),int(t[1]))
            pt2 = (int(t[2]),int(t[3]))
            pt3 = (int(t[4]),int(t[5]))
            ps=[pt1,pt2,pt3]
            self.trangleList.append(ps)
            d1=u.dis(pt2,pt3)
            d2=u.dis(pt1,pt3)
            d3=u.dis(pt1,pt2)
            ds=np.array([d1,d2,d3])
            pt=np.argmax(ds)
            last=(pt-1)%3
            next=(pt+1)%3

            ps_last_i = self.vertexs.index(ps[last])
            ps_next_i = self.vertexs.index(ps[next])
            ps_pt_i   = self.vertexs.index(ps[pt])

            delps.append([ps_last_i, ps_next_i])

            self.adjs[ps_last_i][ps_pt_i] = self.adjs[ps_pt_i][ps_last_i] = 1
            self.adjs[ps_pt_i][ps_next_i] = self.adjs[ps_next_i][ps_pt_i] = 1

        for p in delps:
            s = p[0]
            e = p[1]
            self.adjs[s][e] = self.adjs[e][s] = -1
    
    #Calculate the structural weight of the connected edge
    def lineconf(self,e):
        start=self.vertexs[e[0]]
        end=self.vertexs[e[-1]]
        x1,y1=start
        x2,y2=end
        n=math.floor(u.dis(start,end))
        xlist=np.linspace(start = x1, stop =x2, num = n, dtype = int)
        ylist=np.linspace(start = y1, stop =y2, num = n, dtype = int)
        
        conf=0.0
        for i in range(n):
            conf+=min(self.edgeconf[ylist[i]][xlist[i]]+self.regionconf[ylist[i]][xlist[i]],1)
        conf/=n

        return 1-conf
    
    #Subgraph optimization based on structural weights
    def deledge(self):
        edegs=[]
        dis=dict()
        for i,e in enumerate(self.edges):
            start=self.vertexs[e[0]]
            end=self.vertexs[e[2]]
            # dis['{}'.format(i)]=u.dis(start,end)
            dis['{}'.format(i)]=self.lineconf(e)
        d_order=sorted(dis.items(),key=lambda x:x[1],reverse=False)
        
        for d in d_order:
            e=self.edges[int(d[0])]
            start=self.vertexs[e[0]]
            end=self.vertexs[e[2]]
            
            if d[1]<MIN_EDGE_CONF and u.dis(start,end) > MAX_DEL_DIS and len(self.nearps(e[0]))>2 and len(self.nearps(e[2]))>2:
                self.adjs[e[0]][e[2]]=self.adjs[e[2]][e[0]]=0
            else:
                edegs.append(e)
        self.edges=edegs                            
 
    #Create an edge set
    def build_edges(self):
        self.edges=[]
        for i in range(self.vnum):
            for j in range(i+1,self.vnum):
                if self.adjs[i][j]==1:
                    self.edges.append([i,1,j])
                
    #visualize processed img
    def draw_img(self):
        self.draw_delaunay()
        self.delsingle()
        self.build_edges()
        self.deledge()
        D=Graph_Optimize(self.adjs,self.vertexs,self.name)
        self.vertexs=D.vertexs
        self.edges=D.edges
        self.adjs=D.adj
        self.delsmallrec()
        optimize_vertexs=[]
        can=np.zeros((segsize,segsize,3))
        self.edges=np.argwhere(self.adjs>0)
        self.edges=[[self.vertexs[e[0]],self.vertexs[e[-1]]] for e in self.edges]
        for e in self.edges:
                conf=1
                color=conf*255
                cv2.line(can,e[0],e[-1],(color,color,0),2)
                for p in e:
                    if not p in optimize_vertexs:
                        optimize_vertexs.append(p)
                
        self.vertexs=optimize_vertexs
        for p in self.vertexs:
            cv2.circle(can,p,2,(0,255,255),-1)

        
        self.data=dict()
        self.edges=[[self.vertexs.index(e[0]),self.vertexs.index(e[-1])] for e in self.edges]
        self.data['corners']=np.array(self.vertexs)
        self.data['edges']=np.array(self.edges)

        self.edges=[[self.vertexs[e[0]],self.vertexs[e[-1]]] for e in self.edges]
        self.can=can

    #Get the neighbor node of the node
    def nearps(self,i):
        adj=self.adjs[i]
        pn=np.argwhere(adj>0)
        pn=[n[0] for n in pn]
        return pn
    
    #delete hanging points and cut points
    def delsingle(self):
        for i in range(self.vnum):
            ps=self.nearps(i)
            if len(ps)==1:
                last=i
                cur=int(ps[0])
                self.adjs[cur][last]=self.adjs[last][cur]=0
                while(len(self.nearps(cur))<2):
                    ps=self.nearps(cur)
                    cur_=int(ps[0])
                    last=cur
                    cur=cur_
                    self.adjs[cur][last]=self.adjs[last][cur]=0
        after_del_points=[]
        for i in range(self.vnum):
            if len(self.nearps(i))>=2:
                after_del_points.append(self.vertexs[i])
        vnum=len(after_del_points)
        adj=np.zeros((vnum,vnum))
        for i in range(vnum):
            point=after_del_points[i]
            ori_i=self.vertexs.index(point)
            ad=np.argwhere(self.adjs[ori_i]>0)
            for p in ad:
                p_id=after_del_points.index(self.vertexs[p[0]])
                adj[i][p_id]=adj[p_id][i]=1

        self.vertexs=after_del_points
        self.vnum=vnum
        self.adjs=adj

    #postprocess
    def delsmallrec(self):
        MINEDGE=20
        MAXANGLE=150
        def isdelvertex(v):
            if self.adjs.sum(0)[v]==2 and self.deladj.sum(0)[v]==2:
                for ner in self.nearps(v):
                    if len(self.nearps(ner))<3 and not self.deladj.sum(0)[ner]==2:
                        return False                        
                return True
            else:
                return False
        
        def deladj(p1,p2,adj=False,deladj=False):
            if adj:
                self.adjs[p1][p2]=self.adjs[p2][p1]=0
            if deladj:
                self.deladj[p1][p2]=self.deladj[p2][p1]=0
        
        def delvertex(v):
            ners=self.nearps(v)
            for n in ners:
                deladj(v,n,adj=True)
                deladj(v,n,deladj=True)
        
        
        self.deladj=np.zeros(self.adjs.shape)
        edges=[[self.vertexs.index(e[0]),self.vertexs.index(e[-1])] for e in self.edges]
        for e in edges:
            if u.dis(self.vertexs[e[0]],self.vertexs[e[-1]])<MINEDGE:
                e1_ners=self.nearps(e[0])
                e1_ners.remove(e[-1])
                e2_ners=self.nearps(e[-1])
                e2_ners.remove(e[0])
                for ner0 in e1_ners:
                    for ner1 in e2_ners:
                        if self.adjs[ner0][ner1] and \
                            u.dis(self.vertexs[e[0]],self.vertexs[ner0])<MINEDGE and \
                            u.dis(self.vertexs[e[-1]],self.vertexs[ner1])<MINEDGE and \
                            u.dis(self.vertexs[ner0],self.vertexs[ner1])<MINEDGE:
                            
                            if self.deladj[e[0]][e[-1]]==0 or self.deladj[e[0]][ner0]==0 or self.deladj[e[-1]][ner1]==0 or self.deladj[ner0][ner1]==0:
                                self.deladj[e[0]][e[-1]]=self.deladj[e[-1]][e[0]]=1 if self.deladj[e[0]][e[-1]]==0 else -1
                                self.deladj[e[0]][ner0]=self.deladj[ner0][e[0]]=1 if self.deladj[e[0]][ner0]==0 else -1
                                self.deladj[e[-1]][ner1]=self.deladj[ner1][e[-1]]=1 if self.deladj[e[-1]][ner1]==0 else -1
                                self.deladj[ner0][ner1]=self.deladj[ner1][ner0]=1 if self.deladj[ner0][ner1]==0 else -1
        es=np.argwhere(self.deladj==-1)
        for e in es:
            self.adjs[e[0]][e[-1]]=self.adjs[e[-1]][e[0]]=0
            for p in e:
                nearps=self.nearps(p)
                if len(nearps)==2:
                    deladj(p,nearps[0],adj=True)
                    deladj(p,nearps[1],adj=True)
                    self.adjs[nearps[0]][nearps[1]]=self.adjs[nearps[1]][nearps[0]]=1
                    deladj(p,nearps[0],deladj=True)
                    deladj(p,nearps[1],deladj=True)
                    self.deladj[nearps[0]][nearps[1]]=self.deladj[nearps[1]][nearps[0]]=1
        
        self.delv=[]         
        es=np.argwhere(self.deladj.sum(0)==2)
        for e in es:
            e=e[0]
            if isdelvertex(e):
                ners=self.nearps(e)
                self.delv.append(e)
                for n in ners:
                    if isdelvertex(n):
                        delvertex(n)
                delvertex(e)
        
        es=np.argwhere(self.adjs.sum(0)==2)
        for e in es:
            e=e[0]
            ners=self.nearps(e)
            if len(ners)>2:
                continue
            ner1,ner2=ners
            angle=u.angle(self.vertexs[ner1],self.vertexs[e],self.vertexs[ner2])
            if angle>=MAXANGLE:           
               deladj(e,ner1,adj=True)
               deladj(e,ner2,adj=True)
               self.adjs[ner1][ner2]=self.adjs[ner2][ner1]=1
                            
#Graph initialization and optimization from points heatmap 
def get_result_from_polygraph(dataname):
    u.makedir(process_dir)
    for impt in tqdm(glob.glob(heatmap_dir+'/*')):
        img=cv2.resize(cv2.imread(impt),(segsize,3*segsize))
        name=impt.split('//')[-1].split('\\')[-1].split('/')[-1]
        graph=Graph(img,name)
        graph.draw_img()
        if IMG_SAVE:
            cv2.imwrite(os.path.join(process_dir,name),graph.can)
        
        npy_pt=os.path.join(npy_dir,'{}'.format(name.split('.png')[0]))
        if dataname=='Structure3d':
            np.save(npy_pt,graph.data)
        elif dataname=='Lianjia':
            pred_data=dict()
            pred_data['corners']=np.array(graph.vertexs)
            edges=[(graph.vertexs.index(e[0]),graph.vertexs.index(e[1])) for e in graph.edges]
            pred_data['edges']=np.array(edges)
            np.save(npy_pt,pred_data)
        else:
            raise Exception('wrong dataname!')


    
if __name__=="__main__":
    get_result_from_polygraph(dataname)
        