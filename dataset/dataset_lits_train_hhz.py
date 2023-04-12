from posixpath import join
import os
import sys
#for path in sys.path:
#    print(path)

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import RandomCrop

#sys.path.append("E:\\hehz_alveolar_bone_segmentation_unet3D\\3DUNet-Pytorch_gpu")
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize,Normalize3D
from config import args
import json

#%%
class Train_Dataset(Dataset):
    def __init__(self, args,data_path,is_transform=False):

        self.args = args

        self.filename_list = self.load_file_name_list(data_path)

        self.is_transform=is_transform
        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])
    
        self.transforms_base = Compose([
                Resize(128,128),
                Normalize3D(),
            ])


    def __getitem__(self, index):

        ct = sitk.ReadImage(os.path.join(args.dataset_path,self.filename_list[index][0]), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(args.dataset_path,self.filename_list[index][1]), sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0) # resize to (1,256,256,256)
        
#        print(ct_array.shape,seg_array.shape)
        ct_array,seg_array = self.transforms_base(ct_array,seg_array)
#        print(ct_array.shape,seg_array.shape)

        if self.is_transform:
            ct_array,seg_array = self.transforms(ct_array, seg_array)     

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    
    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as f:
            json_data=json.load(f)
            
        for _ in json_data["training"]:
            file_name_list.append([_['image'][2:],_['label'][2:]])
        return file_name_list

if __name__ == "__main__":
#    sys.path.append('/ssd/lzq/3DUNet')
    args.dataset_path="F:/cbct_data_unet/Task94_Teeth"
    data_path=os.path.join("F:/cbct_data_unet/Task94_Teeth", 'dataset.json')
    train_ds = Train_Dataset(args,data_path)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 6, False, num_workers=0)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())  #torch.Size([2, 1, 256, 256, 256]) torch.Size([2, 256, 256, 256])
    
    '''
    # ---------------- test --------------------
    import matplotlib.pyplot as plt
    nii_path=os.path.join(args.dataset_path, train_ds.filename_list[0][0])
    nii_path="F:\\cbct_data_unet\\Task94_Teeth\\imagesTr\\1000813648_20180116.nii.gz"
    ct = sitk.ReadImage(nii_path, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    plt.imshow(ct_array[100],cmap='gray')
    ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
    
    nii_label_path="F:\\cbct_data_unet\\Task94_Teeth\\labelsTr\\1000813648_20180116.nii.gz"
    ct_label = sitk.ReadImage(nii_label_path, sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(ct_label)
    plt.imshow(seg_array[100],cmap='gray')
    seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

    ct_array = ct_array / args.norm_factor
#    plt.imshow(ct_array[100],cmap='gray')
    print(ct_array.shape,ct_array.unsqueeze(0).shape)  # (512, 512, 512)  torch.Size([1, 512, 512, 512])
    
    #resize=Resize(256,256)
    transforms_base = Compose([
        Resize(256,256),
        Normalize3D(),
    ])

    img, mask=transforms_base(ct_array,seg_array)
    print(img.shape,mask.shape)
    
    plt.imshow(seg_array[0,279],cmap='gray')
    plt.imshow(mask[0,255],cmap='gray')
    
    # -----------------------------------------
    
    
    json_path="F:\\cbct_data_unet\\Task94_Teeth\\dataset.json"
    with open(json_path,'r') as f:
        data=json.load(f)
    
    file_name_list=[]
    for _ in data['training']:
        file_name_list.append([_['image'][2:],_['label'][2:]])
    print(len(file_name_list),file_name_list)
    
    
    #-----------------------------------------
    print(np.unique(seg_array))
    seg_array=torch.LongTensor(seg_array.numpy())
    print(seg_array.dtype,seg_array.shape)
    n, s, h, w = seg_array.size()
    print(n, s, h, w)
    one_hot = torch.zeros(n, 3, s, h, w).scatter_(1, seg_array.view(n, 1, s, h, w), 1) 
    
    
    '''
