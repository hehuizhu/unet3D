from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid,Softmax
import torch
import torch.nn as nn

class UNet3D(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self,in_channels=1,out_channels=3,feat_channels=[64, 128, 256, 512, 1024], residual='conv'):

        #residual: 是否加入残差边，不加则为None
        super(UNet3D, self).__init__()
        self.out_channels=out_channels

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1,2,2))
        self.pool2 = MaxPool3d((1,2,2))
        self.pool3 = MaxPool3d((1,2,2))
        self.pool4 = MaxPool3d((1,2,2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()
        self.Softmax = Softmax(dim=1)

    def forward(self, x):

        # 编码器
        x1 = self.conv_blk1(x)
#        print("x1: ",x1.shape)
        x_low1 = self.pool1(x1)
#        print("x_low1: ",x_low1.shape)
        x2 = self.conv_blk2(x_low1)
#        print("x2: ",x2.shape)
        x_low2 = self.pool2(x2)
#        print("x_low2: ",x_low2.shape)
        x3 = self.conv_blk3(x_low2)
#        print("x3: ",x3.shape)
        x_low3 = self.pool3(x3)
#        print("x_low3: ",x_low3.shape)        
        x4 = self.conv_blk4(x_low3)
#        print("x4: ",x4.shape)
        x_low4 = self.pool4(x4)
#        print("x_low4: ",x_low4.shape)
        base = self.conv_blk5(x_low4)
#        print("base: ",base.shape)

        # 解码器
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
#        print("d4: ",d4.shape)
        d_high4 = self.dec_conv_blk4(d4)
#        print("d_high4: ",d_high4.shape)
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
#        print("d3: ",d3.shape)
        d_high3 = self.dec_conv_blk3(d3)
#        print("d_high3: ",d_high3.shape)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
#        print("d2: ",d2.shape)
        d_high2 = self.dec_conv_blk2(d2)
#        print("d_high2: ",d_high2.shape)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
#        print("d1: ",d1.shape)
        d_high1 = self.dec_conv_blk1(d1)
#        print("d_high1: ",d_high1.shape)
        
        if self.out_channels==1:
            seg = self.sigmoid(self.one_conv(d_high1))
        else:
            seg = self.Softmax(self.one_conv(d_high1),)
#        print("seg: ",seg.shape)

        return seg

    def forward_hhz(self, x):

        # 编码器
        x1 = self.conv_blk1(x)
        print("x1: ",x1.shape)
        x_low1 = self.pool1(x1)
        print("x_low1: ",x_low1.shape)
        x2 = self.conv_blk2(x_low1)
        print("x2: ",x2.shape)
        x_low2 = self.pool2(x2)
        print("x_low2: ",x_low2.shape)
        x3 = self.conv_blk3(x_low2)
        print("x3: ",x3.shape)
        x_low3 = self.pool3(x3)
        print("x_low3: ",x_low3.shape)        
        x4 = self.conv_blk4(x_low3)
        print("x4: ",x4.shape)
        x_low4 = self.pool4(x4)
        print("x_low4: ",x_low4.shape)
        base = self.conv_blk5(x_low4)
        print("base: ",base.shape)

        # 解码器
        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        print("d4: ",d4.shape)
        d_high4 = self.dec_conv_blk4(d4)
        print("d_high4: ",d_high4.shape)
        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        print("d3: ",d3.shape)
        d_high3 = self.dec_conv_blk3(d3)
        print("d_high3: ",d_high3.shape)
        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        print("d2: ",d2.shape)
        d_high2 = self.dec_conv_blk2(d2)
        print("d_high2: ",d_high2.shape)
        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        print("d1: ",d1.shape)
        d_high1 = self.dec_conv_blk1(d1)
        print("d_high1: ",d_high1.shape)
        
        if self.out_channels==1:
            seg = self.sigmoid(self.one_conv(d_high1))
        else:
            seg = self.Softmax(self.one_conv(d_high1))
        print("seg: ",seg.shape)

        return seg


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        #3D反卷积
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel),
                                    stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                        ReLU())

    def forward(self, x):

        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):

        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n,c,d*w*h).permute(0,2,1)
        pooled = self.pool_1d(inp)
        c = int(c/self.kernel_size[0])
        return inp.view(n,c,d,w,h)



#%%
#if __name__=='__main__':
#    # [16, 32, 64, 128, 256] [32, 64, 128, 256, 512]
#    net = UNet3D(1,3)
#    x = torch.ones(2, 1, 128, 128, 128)
#    output=net(x)
#
#    print (output.size())  # torch.Size([2, 3, 128, 256, 256])
        
    
#%%
'''
if __name__=='__main__':
    import os
    import sys
    sys.path.append("E:\\hehz_alveolar_bone_segmentation_unet3D\\3DUNet-Pytorch_gpu")
    from torch.utils.data import DataLoader,random_split
    from config import args
    from dataset.dataset_lits_train_hhz import Train_Dataset
    args.dataset_path="F:/cbct_data_unet/Task94_Teeth"
    data_path=os.path.join("F:/cbct_data_unet/Task94_Teeth", 'dataset.json')
    train_ds = Train_Dataset(args,data_path)

    # 定义数据加载
    train_dl = DataLoader(train_ds, 2, False, num_workers=0)

#    for i, (ct, seg) in enumerate(train_dl):
#        print(i,ct.size(),seg.size())  #torch.Size([2, 1, 256, 256, 256]) torch.Size([2, 256, 256, 256])
#    net = UNet3D(residual=None)

    # [16, 32, 64, 128, 256] [32, 64, 128, 256, 512]
    net = UNet3D(in_channels=1, out_channels=3,feat_channels=[16, 32, 64, 128, 256],residual=None)
    import torch
    x = torch.ones(2, 1, 128, 128, 128)
    
    for i, (ct, seg) in enumerate(train_dl):
        if i==0:
            print(i,ct.size(),seg.size())  #torch.Size([2, 1, 128, 128, 128]) torch.Size([2, 128, 128, 128])
            output=net(ct)
    

    print (output.size())  #[2, 3, 128, 128, 128]
'''
    
#%%
    
'''
3D卷积, 输入的shape是( N , Cin , D , H , W ):

N 就是batch_size；Cin则对应着输入图像的通道数，如RGB\BGR图像这一维度就是3；
D则是深度，如果是对于视频序列使用的3d conv，那么这个D实际对应的就是要执行卷积的frame_size;
H,W对应的就是输入图像的高和宽
'''
'''
if __name__=='__main__':
    conv=nn.Conv2d(1,3,3,1,1)
    
    x=torch.randn(1,1,48,48)
    
    y=conv(x)
    print(y.shape)  #[1, 3, 48, 48]
    
    conv3d=nn.Conv3d(1,64,(2,3,3),(1,1,1),(0,1,1))  # 三个维度上的 conv kernel\步长\padding
    x=torch.randn(1,1,32,48,48)
    
    y=conv3d(x)
    print(y.shape)  # torch.Size([1, 64, 31, 48, 48])
    
    # Sigmoid and softmax
    sigmoid=nn.Sigmoid()
    x=torch.randn(1,1,2,6,6)
    print(x,sigmoid(x))
    
    print("------ softmax ------")
    softmax_nn=nn.Softmax()
    x=torch.randn(2,3,6,6)
    print(x,'\n','-----\n',softmax_nn(x))
    
    # 手动实现 Softmax
    def softmax(x):
        # x.shape: (B,C,H,W)
        a=x.exp()  #torch.e**x
        softmax_value=a/torch.sum(a,dim=1,keepdim=True)
        return softmax_value
    
    softmax_value=softmax(x)
    print(softmax_value)
    
    # 3d  softmax
    x=torch.randn(2,2,2,6,6)
    print('\n','-----\n',softmax_nn(x))
    
    softmax_value=softmax(x)
    print(softmax_value)
'''

#%%

