import os
import numpy as np
import json
import cv2
import math
from scipy.ndimage import gaussian_filter
import random

#create new folder
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#generate hop spot
def blurjunction(points):
    w,h,c=points.shape
    new_can=np.zeros((w,h,c))
    points=np.argwhere(points)
    for p in points:
        can=np.zeros((w,h,c))
        cv2.circle(can,(p[1],p[0]),1,(255,255,255),-1)
        can=blur(can,2)
        new_can=np.where(can>new_can,can,new_can)
    return new_can
        
        
#Judge if the coordinates are in range
def pointexist(p,points):
    for point in points:
        if p[0]==point[0] and p[1]==point[1]:
            return True
    return False

#enlarge image
def expandim(im,exsize):
    img=np.zeros((exsize,exsize,3))
    w,h,c=im.shape
    img[:w,:h,:]=im
    return img

#return angle pa<-pb->pc
def angle(pa,pb,pc):
    a=dis(pb,pc)
    b=dis(pa,pc)
    c=dis(pb,pa)
    if a==0 or c==0:
        return 180
    else:
        cos=(a**2+c**2-b**2)/(2*a*c)
        cos=max(-1,min(1,cos))
        angle=180*math.acos(cos)/math.pi
        return angle

#blur image
def blur(img,sigma):
    img=gaussian_filter(img,sigma)
    maxi = np.amax(img)
    img = img * 255/maxi
    return img

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

#p rotates theta clockwise around point p_
def rotate(p,theta,p_):
    theta=theta*math.pi/180
    x,y=p
    x_,y_=p_
    _x=(x-x_)*math.cos(theta)-(y-y_)*math.sin(theta)+x_
    _y=(y-y_)*math.cos(theta)+(x-x_)*math.sin(theta)+y_
    return (_x,_y)

#Convert points from bound to size 
def scale_points(points,bound,size):
    xmin,xmax,ymin,ymax=bound
    ori_size=max(ymax-ymin,xmax-xmin,size)
    points=[(p[0]-xmin,p[1]-ymin) for p in points]
    points=[(int(p[0]*size/ori_size),int(p[1]*size/ori_size)) for p in points]
    return points


#rotate image and label
# input: 
    # points:position
    # imgs:image of corresponding vertices [im1,im2,...]
    # rotheta:rotate angle
# output:
    # points:position after rotation
    # dst:rotated img [im1,im2,...]
def rotate_sample(points,img,rotheta):
        shape=img[0].shape
        basep=(shape[0]/2,shape[1]/2) 
        points=[rotate(p,rotheta,basep) for p in points]
        vs=np.array(points)
        delta=10
        xmin=min(vs[:,0])-delta
        xmax=max(vs[:,0])+delta
        ymin=min(vs[:,1])-delta
        ymax=max(vs[:,1])+delta
        points=scale_points(points,(xmin,xmax,ymin,ymax),shape[0])
        bound=(xmin,xmax,ymin,ymax)
        
        
        height, width = shape[:2]    
        center = (int(3*width / 2), int(3*height / 2))  
        angle = 360-rotheta 
        scale = 1                        
        
        dst=[]
        for im in img:
            ori_hp=np.zeros((3*height,3*width,3))
            ori_hp[height:-height,width:-width,:]=im
            

            M = cv2.getRotationMatrix2D(center, angle, scale)
        
            im= cv2.warpAffine(src=ori_hp, M=M, dsize=(3*height, 3*width), borderValue=(0, 0, 0))
            M = np.float32([[1, 0, -xmin], [0, 1, 0]])
            im= cv2.warpAffine(src=im, M=M, dsize=(3*height, 3*width), borderValue=(0, 0, 0))
            
            M = np.float32([[1, 0, 0], [0, 1, -ymin]])
            im= cv2.warpAffine(src=im, M=M, dsize=(3*height, 3*width), borderValue=(0, 0, 0))
            
            
            ori_size=max(ymax-ymin,xmax-xmin,height)
            im=im[height:int(height+ori_size),height:int(height+ori_size),:]
            
            dst.append(cv2.resize(im, (height,height)))
        return points,dst

#sample augmentation
def data_augment(image, brightness):
    factor = 1.0 + random.uniform(-1.0*brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0,255).astype(np.uint8)
    image = cv2.LUT(image, table)
    return image
    
#return distance between p1 and p2
def dis(p1,p2):
    return pow(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2),0.5)

#return distance between p(point) and line(se)
def pdisl(p,s,e):
    ang=angle(p,s,e)*math.pi/180
    d=dis(p,s)
    distance=d*math.sin(ang)
    # print(math.acos(0.5),ang,math.sin(ang),p,s,e,distance)
    return distance

#create adj matrix from edges set
def get_adj_from_edges(edges):
    vertexs=[]
    new_edges=[]
    vmap={}
    
    for e in edges:
        if  not e[0]==e[-1] :
            new_edges.append(e)
            for p in e:
                if not p in vertexs:
                    vertexs.append(p)
    for i,p in enumerate(vertexs):
        vmap['{}_{}'.format(p[0],p[1])]=i
    adj=np.zeros((len(vertexs),len(vertexs)))
    for e in edges:
        i=vmap['{}_{}'.format(e[0][0],e[0][1])]   
        j=vmap['{}_{}'.format(e[1][0],e[1][1])]
        adj[i][j]=adj[j][i]=1
    return vertexs,new_edges,adj,vmap

#return coutour of mask
def polygonize_mask(mask, degree=0.01, return_mask=False):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask

    room_mask = 255 * (mask == 1)
    room_mask = room_mask.astype(np.uint8)
    room_mask_inv = 255 - room_mask

    ret, thresh = cv2.threshold(room_mask_inv, 250, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    perimeter = cv2.arcLength(cnt, True)
    # epsilon = 0.01 * cv2.arcLength(cnt, True)
    epsilon = degree * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # approx = np.concatenate([approx, approx[0][None]], axis=0)
    approx = approx.astype(np.int32).reshape((-1, 2))

    # approx_tensor = torch.tensor(approx, device=self.device)

    # return approx_tensor
    if return_mask:
        room_filled_map = np.zeros((h, w))
        cv2.fillPoly(room_filled_map, [approx], color=1.)

        return approx, room_filled_map
    else:
        return approx
    

