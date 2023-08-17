import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2
import glob
import torchvision.transforms as transforms
import random



class HMDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir,isval,im_size):
        self.samples=glob.glob(img_dir+'/*')
        self.names=os.listdir(img_dir)
        self.im_size=im_size
        self.inputs=[]
        self.outputs_room=[]
        self.outputs_wall=[]
        self.outputs_points=[]
        self.outputs_junctions=[]
        self.isval=isval
        for ims in self.samples:
            im=cv2.imread(ims)
            try:
                self.inputs.append(im[:im_size,:,:])
                self.outputs_room.append(im[im_size:2*im_size,:,0])
                self.outputs_points.append(im[2*im_size:3*im_size,:,0])
                self.outputs_junctions.append(im[2*im_size:3*im_size,:,1])
                self.outputs_wall.append(im[-im_size:,:,0])
            except:
                print(im)
            
            
    def __len__(self):
        return len(self.inputs)
    
    def process(self,im):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=2)
        im=im.transpose((2, 0, 1))
        return im
    
    def __getitem__(self, idx):
        input=self.inputs[idx]
        im_size=self.im_size
        output_room=self.outputs_room[idx]
        output_wall=self.outputs_wall[idx]
        output_points=self.outputs_points[idx]
        output_junctions=self.outputs_junctions[idx]
        # x=random.randint(0,100)/100
        # degree=random.randint(0,9)*10
        if not self.isval:
            p1 = random.randint(0,1)
            p2 = random.randint(0,1)
            trans= transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(im_size, padding=20),
                transforms.Resize((im_size, im_size)),
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2)
            ])
            trans_input=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((im_size, im_size)),
            ])
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            input=np.array(trans(input))
            input=np.array(trans_input(input))
            torch.random.manual_seed(seed)
            output_room=np.array(trans(output_room))
            torch.random.manual_seed(seed)
            output_wall=np.array(trans(output_wall))
            torch.random.manual_seed(seed)
            output_points=np.array(trans(output_points))
            torch.random.manual_seed(seed)
            output_junctions=np.array(trans(output_junctions))
        else:
            input,output_room,output_wall,output_points,output_junctions=map(lambda t:cv2.resize(t, (im_size,im_size), interpolation=cv2.INTER_LINEAR),(input,output_room,output_wall,output_points,output_junctions))
        name=self.names[idx]
        input=self.process(input)
        output_room=self.process(output_room) 
        output_wall=self.process(output_wall)
        output_points=self.process(output_points)  
        output_junctions=self.process(output_junctions)      
        sample = {'input': torch.from_numpy(input).float(),
                'output_room': torch.from_numpy(output_room).float().div(255.0),
                'output_wall': torch.from_numpy(output_wall).float().div(255.0),
                'output_points': torch.from_numpy(output_points).float().div(255.0),
                'output_junctions': torch.from_numpy(output_junctions).float().div(255.0),
                'name':name}
        return sample

def get_dataset(val ,img_dir, batch_size,im_size):

    dataset = HMDataset(img_dir,val,im_size)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=not val,
                                            num_workers=0)
    if val:
        data_loaders = {'val': dataloader}
        dataset_sizes = {}
        dataset_sizes['val'] = len(dataset)
    else:
        data_loaders = {'train': dataloader}
        dataset_sizes = {}
        dataset_sizes['train'] = len(dataset)

    return data_loaders, dataset_sizes
