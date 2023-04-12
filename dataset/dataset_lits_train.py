from posixpath import join
import os
import sys
for path in sys.path:
    print(path)
    
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import RandomCrop

sys.path.append(os.getcwd())
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize
from config import args

#%%
class Train_Dataset(Dataset):
    def __init__(self, args,is_transform=False):

        self.args = args

        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))

        self.is_transform=is_transform
        self.resize=Resize(256,256)
        self.transforms = Compose([
                RandomCrop(self.args.crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                # RandomRotate()
            ])

    def __getitem__(self, index):

        ct = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
        seg = sitk.ReadImage(self.filename_list[index][1], sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        ct_array = self.resize(torch.FloatTensor(ct_array).unsqueeze(0))
        seg_array = self.resize(torch.FloatTensor(seg_array).unsqueeze(0)) # resize to (1,256,256,256)

        if self.is_transform:
            ct_array,seg_array = self.transforms(ct_array, seg_array)     

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  # 整行读取数据
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
#    sys.path.append('/ssd/lzq/3DUNet')
    train_ds = Train_Dataset(args)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i,ct.size(),seg.size())
    
    '''
    # ---------------- test --------------------
    import matplotlib.pyplot as plt
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
    
    resize=Resize(256,None)
    img, mask=resize(ct_array,seg_array)
    print(img.shape,mask.shape)
    
    plt.imshow(seg_array[0,279],cmap='gray')
    plt.imshow(mask[0,279],cmap='gray')
    
    # -----------------------------------------
    '''
    
    
    
