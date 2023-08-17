import numpy as np
from utils import utils as u
import os
import glob
import cv2

Threshold=3 #scenecad
AngleThreshold=150




def get_line(start,first_ner,adjs,points):
    lines=[start,first_ner]
    ner=lines[-1]
    last=lines[0]
    degree=len(np.argwhere(adjs[ner]))
    while(degree<3 and ner!=lines[0]):
        cur=ner
        try:
            ner=np.argwhere(adjs[ner])[0][0] if not np.argwhere(adjs[ner])[0][0]==last else np.argwhere(adjs[ner])[1][0]
        except:
            return None
        lines.append(ner)
        degree=len(np.argwhere(adjs[ner]))
        last=cur

    return lines


# Graph optimization
class Graph_Optimize(object):
    def __init__(self,adjs,points,name):
        self.ori_adjs=adjs
        self.ori_points=points
        self.name=name
        # self.optimize_junction()
        
        self.edges=self.optimize_graph(self.ori_adjs,self.ori_points,Threshold)
        self.vertexs,self.edges,self.adj,self.vmap=u.get_adj_from_edges(self.edges)
        
        
    #Using Douglas algorithm to optimize the whole structure
    def optimize_graph(self,adjs,points,threshold):
        search=[]
        junctions=np.squeeze(np.argwhere(np.sum(adjs,1)>2))
        if len(np.argwhere(np.sum(adjs,1)>2))==1:
            junctions=[junctions]
        if len(junctions)==0:
            junctions=[0]
        edges=[]
        for jun in junctions:
            ners=np.squeeze(np.argwhere(adjs[jun]>0))
            for n in ners:
                if not n in search:
                    lines_id=get_line(jun,n,adjs,points)
                    lines=[points[id] for id in lines_id]
                    # if len(junctions)==1:
                    #     print(lines_id)
                    #     assert 0
                    d=Douglas(lines,threshold)
                    edges.extend(d.edges)
                    search.extend(lines_id[1:-1])
                    if not lines_id[-1] in junctions:
                        search.append(lines_id[-1])
        return edges
              
    def get_ners(self,id,adj):
        ners=[n[0] for n in np.argwhere(adj[id]>0)]
        return ners
            
    def get_edge(self):
        self.edges=[]
        for adj in np.argwhere(self.adj>0):
            if adj[0]<adj[1]:
                self.edges.append([adj[0],adj[1]])
        
class Douglas(object):
    def __init__(self,points,threshold):
        self.threshold=threshold
        self.points=points
        # approx = cv2.approxPolyDP(self.points, 20, True)
        self.vertexs=np.zeros(len(self.points))
        self.vertexs[0]=self.vertexs[-1]=1
        self.edges=[]
        self.generate_edge()
        
    def skin(self,left,right):
        id=self.get_vertex(left,right)
        if id==None:
            return 
        else:
            self.vertexs[id]=1
            self.skin(left,id)
            self.skin(id,right)
    
    def get_vertex(self,left,right):
        max=0
        maxid=None
        if len(self.points)>2 and self.points[left]==self.points[right]:
            line_left=self.points[left]
            line_right=self.points[right-1]
        else:
            line_left=self.points[left]
            line_right=self.points[right]
        # max_dis=self.threshold*cv2.arcLength(np.array([self.points[id] for id in range(left,right+1)]),False)

        for i in range(left+1,right):
            # if self.points[i] in [self.points[left],self.points[right]] or self.points[left]==self.points[right]:
            #     continue
            # try:
            dis=u.pdisl(self.points[i],line_left,line_right)
            if dis>max:
                max=dis
                maxid=i
            # except:
            #     print(self.points[i],self.points[left],self.points[right])
            #     print(self.points[i] in [self.points[left],self.points[right]])
        if max>=self.threshold:
            return maxid
        else:
            return None
        
    def generate_edge(self):
        self.skin(0,len(self.points)-1)
        start=self.points[0]
        for i in np.argwhere(self.vertexs[1:]>0):
            end=self.points[i[0]+1]
            self.edges.append([start,end])
            start=end
        
            