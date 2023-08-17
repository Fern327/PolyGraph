
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage

class FocalLossV1(nn.Module):
    
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, label,logits):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        #将输出结果归一化
        probs = torch.sigmoid(logits)
        #label==1的地方用probs代替，不等于1的地方用1 - probs代替
        pt = torch.where(label == 1, probs, 1 - probs)
        #Crit是BCEloss
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

 
class AWing(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)
    def forward(self, y_pred , y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
     
        return lossMat


class Loss_weighted(nn.Module):
    def __init__(self,device, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWing(alpha, omega, epsilon, theta)
        self.device=device
 
    def forward(self, y_pred , y):
        M = torch.zeros_like(y)
        M=self.generate_weight_map(M,y)
        M = M.float()
        Loss = self.Awing(y_pred,y)
        weighted = Loss * (self.W * M + 1.)
        return weighted.mean()

    def generate_weight_map(self,weight_map,heatmap):
        weight_map,heatmap=map(lambda t:t.cpu().detach().numpy().copy(),(weight_map,heatmap))
        k_size = 15
        b,c,h,w=heatmap.shape
        weight_map_new=np.zeros((b,c,h,w))
        for i in range(b):
            w,h=map(lambda t:t[i,0,:,:],(weight_map,heatmap))
            dilate = ndimage.grey_dilation(h ,size=(k_size,k_size))
            w[np.where(h>0)] = 1
            weight_map_new[i,0,:,:]=w
            
        weight_map=torch.from_numpy(weight_map_new).float().to(self.device)
        return weight_map

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
        # self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, pred, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = pred.shape[0]

        # target_onehot = self.to_one_hot(target, self.n_classes)
        pred=F.sigmoid(pred)
        # Numerator Product
        inter = pred * target
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, -1).sum(1)

        # Denominator
        # union = pred + target - (pred * target)
        union = pred + target
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, -1).sum(1)

        loss = 2*inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()
