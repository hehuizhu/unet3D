"""
基于Dice的loss函数，计算时pred和target的shape必须相同，亦即target为onehot编码后的Tensor
"""

import torch
import torch.nn as nn

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt



#%%
nii_label_path="F:\\cbct_data_unet\\Task94_Teeth\\labelsTr\\1000813648_20180116.nii.gz"
ct_label = sitk.ReadImage(nii_label_path, sitk.sitkUInt8)
seg_array = sitk.GetArrayFromImage(ct_label)
plt.imshow(seg_array[100],cmap='gray')
print(np.unique(seg_array))
seg_array=torch.LongTensor(seg_array)
print(seg_array.dtype,seg_array.shape)
n=1
s, h, w = seg_array.size()
print(n, s, h, w)
target = torch.zeros(n, 3, s, h, w).scatter_(1, seg_array.view(n, 1, s, h, w), 1) 
print(target.shape)

pred=torch.randn(1,3, s, h, w)
softmax_nn=nn.Softmax(dim=1)
pred=softmax_nn(pred)

torch.clamp(torch.tensor(0.05),0,1)
#%%

a=torch.tensor([[[[[0,1,0],
                   [0,1,0],
                   [1,1,0]],
                    [[0,1,0],
                   [0,1,0],
                   [1,1,0]]],
                [[[0,1,0],
                   [0,1,0],
                   [1,1,0]],
                    [[0,1,0],
                   [0,1,0],
                   [1,1,0]]],
                [[[0,1,0],
                   [0,1,0],
                   [1,1,0]],
                    [[0,1,0],
                   [0,1,0],
                   [1,1,0]]]],
                [[[[0,0,0],
                   [0,1,0],
                   [1,1,0]],
                    [[0,1,0],
                   [0,0,0],
                   [1,1,0]]],
                [[[0,1,0],
                   [0,2,0],
                   [1,1,0]],
                    [[0,1,0],
                   [0,1,0],
                   [1,2,0]]],
                [[[0,1,0],
                   [0,1,0],
                   [1,1,0]],
                    [[0,1,1],
                   [0,1,0],
                   [0,1,2]]]]])

print(a[:,0],'\n','\n',a[:,0].shape) # [2, 2, 3, 3])
print(a[:,0].sum(dim=1),'\n','\n',a[:,0].sum(dim=1).shape)
print(a[:,0].sum(dim=1).sum(dim=1),'\n','\n',a[:,0].sum(dim=1).sum(dim=1).shape)
print(a[:,0].sum(dim=1).sum(dim=1).sum(dim=1),'\n','\n',a[:,0].sum(dim=1).sum(dim=1).sum(dim=1).shape)

# 加权
alpha=torch.tensor([0.5,0.2,0.3])
alpha1=alpha[a.view(-1,1).data.view(-1)] 
print(alpha1.view(a.shape))
