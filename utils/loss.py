"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn

# 重叠度衡量  F1_score的高度抽象
class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)

class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)

# 组合损失  Dice loss + BCEloss
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()  #交叉熵损失函数
        self.bce_weight = 1.0

    def forward(self, pred, target):

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight

# Jaccard 与 Dice 计算方式不同
class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard  += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                        target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)

class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:,i] - target[:,i]).pow(2) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target[:,i].sum(dim=1).sum(dim=1).sum(dim=1))

            s2 = ((pred[:,i] - target[:,i]).pow(2) * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


#   Dice Loss只是Tversky loss的一种特殊形式而已，alpha 和 beta 均为0.5的时候，这个公式就是Dice系数，当 alpha 和 beta 均为1的时候，这个公式就是Jaccard系数。
#  define:TL(p,p')=(p&p')/(p&p'+b*((1-p)&p')+(1-b)*(p&(1-p')))
class TverskyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)+
                        0.3 * (pred[:,i] * (1 - target[:,i])).sum(dim=1).sum(dim=1).sum(dim=1) + 0.7 * ((1 - pred[:,i]) * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


# ------------- 230309 hehz added -----------------
# Focal loss  
# Focal Loss来解决难易样本数量不平衡,gamma用来控制易分样本和难分样本的权重，alpha用来控制正负样本的权重。
# Focal loss对于分类不准确的样本，损失没有改变，对于分类准确的样本，损失会变小 gamma=0.5/1/2/5,
class Focal_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,target,alpha=[0.5,0.5],gamma=2):
        # alpha: list  每一个类的权重因子
        
        fcloss=0.
        if pred.size(1)==1:
            fcloss=(-alpha[0]*((1-pred[:,0]).pow(gamma))*torch.log(pred[:,0])*target[:,1]).sum(dim=1).sum(dim=1).sum(dim=1)
            +(-alpha[1]*(pred[:,0].pow(gamma))*torch.log(1-pred[:,0])*(1-target[:,1])).sum(dim=1).sum(dim=1).sum(dim=1)
            fcloss/=pred.size(2)*pred.size(3)
            return torch.clamp( fcloss.mean(), 0, 2)
        else:
            for i in pred.size(1):
                fcloss+=(-alpha[i]*((1-pred[:,i]).pow(gamma))*torch.log(pred[:,i])*target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)
            
            fcloss/=pred.size(2)*pred.size(3)*pred.size(4)
            return torch.clamp(fcloss.mean(), 0, 2)
    
# BCLOSS  二分类交叉熵与多分类交叉熵  
class BCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,pred,target):
        bcloss=0.
        if pred.size(1)==1:
            bcloss=(-torch.log(pred[:,0])*target[:,1]).sum(dim=1).sum(dim=1).sum(dim=1)+(-torch.log(1-pred[:,0])*(1-target[:,1])).sum(dim=1).sum(dim=1).sum(dim=1)
            bcloss/=pred.size(2)*pred.size(3)
            return torch.clamp( bcloss.mean(), 0, 2)
        else:
            for i in pred.size(1):
                bcloss+=(-torch.log(pred[:,i])*target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)
                
            bcloss/=pred.size(2)*pred.size(3)*pred.size(4)
            return torch.clamp(bcloss.mean(), 0, 2)
        
    
# Balance_BCLOSS  二分类交叉熵与多分类交叉熵    对正负样本加权，对小样本的类别适当加大权重
class BalanceBCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,pred,target,weight=[0.5,0.5]):
        # weight: list , the sum is equal to 1.
        bcloss=0.
        if pred.size(1)==1:
            bcloss=weight[0]*((-torch.log(pred[:,0])*target[:,1]).sum(dim=1).sum(dim=1).sum(dim=1))+weight[1]*(-torch.log(1-pred[:,0])*(1-target[:,1])).sum(dim=1).sum(dim=1).sum(dim=1)
            bcloss/=pred.size(2)*pred.size(3)*pred.size(4)
            return torch.clamp( dice.mean(), 0, 2)
        else:
            for i in pred.size(1):
                bcloss+=weight[i]*((-torch.log(pred[:,i])*target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1))
                
            bcloss/=pred.size(2)*pred.size(3)*pred.size(4)
            return torch.clamp(bcloss.mean(), 0, 2)    
   
# -----------------------------------------------