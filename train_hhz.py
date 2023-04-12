import os
import sys
sys.path.append("E:\\hehz_alveolar_bone_segmentation_unet3D\\3DUNet-Pytorch")

import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader,random_split
import torch
import torch.optim as optim

import logging
from tqdm import tqdm
#from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train_hhz import Train_Dataset
from models import UNet, ResUNet , KiUNet_min, SegNet,UNet3D
from utils import logger, weights_init, metrics, common, loss
import config

#%%

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

# 
def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
#        print(data.shape,target.shape)
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

#        print(data.shape,target.shape)
        output = model(data)
#        print('output: ',output.shape)
#        loss0 = loss_func(output[0], target)
#        loss1 = loss_func(output[1], target)
#        loss2 = loss_func(output[2], target)
#        loss3 = loss_func(output[3], target)
#
#        loss = loss3  +  alpha * (loss0 + loss1 + loss2)
        loss=loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        #train_loss.update(loss3.item(),data.size(0))
        #train_dice.update(output[3], target)
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels==3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./logs', args.save)
    if not os.path.exists(save_path): os.makedirs(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data info  8:2
    data_path=os.path.join(args.dataset_path, 'dataset.json')
    dataset=Train_Dataset(args,data_path)
    n_val=int(args.valid_rate*len(dataset))
    n_train=int(len(dataset)-n_val)  #50 12
    train_set,val_set=random_split(dataset,lengths=[n_train,n_val])
    
    train_loader = DataLoader(dataset=train_set,batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=val_set,batch_size=1,num_workers=args.n_threads, shuffle=False)

    # model info
    '''
    model = ResUNet(in_channel=1, out_channel=args.n_labels,training=True).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    '''
    
    # feat_channels=[16, 32, 64, 128, 256]   32, 64, 128, 256, 512
    if args.model=="UNet3D":
        model = UNet3D(in_channels=args.n_channels, out_channels=args.n_labels,feat_channels=[16, 32, 64, 128, 256],residual=None).to(device)
    if args.model=="ResUNet3D":
        model = UNet3D(in_channels=args.n_channels, out_channels=args.n_labels,feat_channels=[16, 32, 64, 128, 256],residual='conv').to(device)
    else:
        print("The model type must be 'UNet3D' ! ")

#    model=model.cuda()
    if args.optimizer=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        pass
    
    common.print_network(model)    
 
    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path,"train_log")
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"Using device {device}")
    logging.info(f'Network:\n'
             f'\t{n_train} tainset number\n'
             f'\t{n_val} valset number\n'
             f'\t{args.model} model\n'
             f'\t{args.batch_size} batch size\n'
             f'\t{args.n_channels} input channels\n'
             f'\t{args.n_labels} output channels (classes)\n'
             f'\t{args.epochs} epochs\n'
             f'\t{args.lr} learning rate\n'
             f'\t{args.optimizer} optimizer\n')


    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    epoch=1
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        # Save 验证集效果最好的 模型
        if val_log['Val_dice_liver'] > best[1]:
            logging.info('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
            logging.info('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping  提前终止策略，如果后面 30epoch的验证集Dice值均未大于最好值，则终止训练
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                logging.info("=> early stopping")
                break
        torch.cuda.empty_cache()    