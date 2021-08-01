#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import os
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# from data_load_gray_multiscale import MouseKeypointsDataset
from data_load_RGB_multiscale import MouseKeypointsDataset
from utilis.checkpointsave import savecheckpoint, saveepochcheckpont, saveepochcheckpont_PCK
from utilis.netparameter160 import config

from utilis.metrics import PCK
# from AttentionCRFmodel.mainStackedHourGlass_full_finalcas2_C1GM2 import *
from AttentionCRFmodel.mainStackedHourGlass_full_finalcas2_C3GM2 import *

from tensorboardX import SummaryWriter

cudnn.benchmark = True

device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")


#DeepLabCut dataset
# netname = 'mousefullnet_finalcas2_C1GM2_4fulltrain'
# validresultname = 'results/mouse/compare/results/validresult_fullnet_finalcas2_C1GM2_4fulltrain.csv'
# trainresultname = 'results/mouse/compare/results/trainresult_fullnet_finalcas2_C1GM2_4fulltrain.csv'



#PDMB
netname = 'pdmousefullnet_finalcas2_C3GM2'
validresultname = 'results/pdmouse/compare/results/validresult_fullnet_finalcas2_C3GM2.csv'
trainresultname = 'results/pdmouse/compare/results/trainresult_fullnet_finalcas2_C3GM2.csv'




#zebra dataset
# netname = 'zebrafullnet_finalcas2_C3GM2'
# validresultname = 'results/zebra/compare/results/validresult_fullnet_finalcas2_C3GM2.csv'
# trainresultname = 'results/zebra/compare/results/trainresult_fullnet_finalcas2_C3GM2.csv'



parser = argparse.ArgumentParser(description='PyTorch Mousepose Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


def get_peak_points(heatmaps):
    """
    heatmap to keypoints
    :param heatmaps: numpy array (N,4,240,320)
    :return:numpy array (N,4,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred_points,gts,indices_valid=None):
    """

    :param pred_points: numpy (N,4,2)
    :param gts: numpy (N,4,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = torch.from_numpy(pred_points).float()
    gts = torch.from_numpy(gts).float()
    criterion = nn.MSELoss()
    corloss = criterion(pred_points,gts)
    rmse = np.sqrt(corloss)
    #print("loss-----: %f" % corloss)
    return rmse

def calculate_mask(heatmaps_targets):
    """
    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N,C,_,_ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    NaN_N_idx = []
    NaN_C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n,c,:,:].max()
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
            else:
                NaN_N_idx.append(n)
                NaN_C_idx.append(c)

    mask = torch.zeros(heatmaps_targets.size())
    mask[N_idx,C_idx,:,:] = 1.
    # mask = mask.float().to(device)
    return mask,[N_idx,C_idx],[NaN_N_idx,NaN_C_idx]



def makenotvalidpointsasnan(predicted, gts,indices_notvalid):
    for i,j in zip(indices_notvalid[0],indices_notvalid[1]):
        predicted[i,j,:] = [-1,-1]
        gts[i, j, :] = [-1, -1]
    return  predicted, gts



def resume_checkpoint(net):
    '''
    :return:
    '''
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join('./checkpoint', netname, 'epoch_6checkpoint.ckpt'))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        config['start_epoch'] = start_epoch+1


def initnet():
    pprint.pprint(config)
    torch.manual_seed(0)
    net = Gmscenet( nChannels=128, nStack=config['nstack'], nModules=1, numReductions=3, nJoints=4)


    # using 2 Gpus
    print("Let's use", torch.cuda.device_count(), "GPUs")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        net = torch.nn.DataParallel(net)
    net.to(device)
    print(device)


    trainDataset = MouseKeypointsDataset(csv_file=config['trainsample'], root_dir='')
    sample_num = len(trainDataset)
    config['train_num'] = sample_num

    validDataset = MouseKeypointsDataset(csv_file=config['validsample'], root_dir='')
    sample_num = len(validDataset)
    config['valid_num'] = sample_num
    print('train:',  config['train_num'], 'validation:',  config['valid_num'])


    # split train and valid
    # train_size = int(0.8 * sample_num)
    # valid_size = sample_num - train_size
    # config['train_num'], config['valid_num'] = train_size, valid_size
    # train_db, val_db = torch.utils.data.random_split(trainDataset, [train_size, valid_size])
    # print('train:', len(train_db), 'validation:', len(val_db))

    trainDataLoader = DataLoader(trainDataset, config['batch_size'], shuffle=True, num_workers=2)
    validDataLoader = DataLoader(validDataset, config['batch_size_valid'], shuffle=True, num_workers=2)

    # if (config['checkout'] != ''):
    #     net.load_state_dict(torch.load(config['checkout'],map_location='cpu'))
    resume_checkpoint(net)
    return net, trainDataLoader,validDataLoader


def train(net, trainDataLoader, epoch):
    train_loss=0
    train_rmse=0
    train_PCK= 0
    train_PCK2= [0, 0, 0, 0 ,0]
    pckcount = [0, 0, 0, 0, 0]
    net.train()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    for i, data in enumerate(trainDataLoader):
        inputs = data['image']
        heatmaps_targets = data['heatmaps']
        heatmaps_targets_2s = data['heatmaps_2s']
        gts = data['keypoints']
        mask, indices_valid, indices_notvalid = calculate_mask(heatmaps_targets)

        heatmaps_targets = heatmaps_targets * mask

        #2s
        mask_2s, indices_valid_2s, indices_notvalid_2s = calculate_mask(heatmaps_targets_2s)
        heatmaps_targets_2s = heatmaps_targets_2s * mask_2s

        inputs = inputs.to(device)
        heatmaps_targets = heatmaps_targets.to(device)
        mask = mask.float().to(device)


        #2s
        heatmaps_targets_2s = heatmaps_targets_2s.to(device)
        mask_2s = mask_2s.float().to(device)

        optimizer.zero_grad()
        outputs, outputs_2s = net(inputs)
        if(i ==0):
            print("outputs:   ", len(outputs))
            print("outputs_2s:   ", len(outputs_2s))

        #intermideate supervision
        loss=0
        loss_gm=0
        loss_2s=0
     
        #
        for j in range(config['nstack']+2):
            stackoutput = outputs[j] * mask
            loss += criterion(stackoutput, heatmaps_targets)
            # 2s
            if (j < config['nstack']+1):
                stackoutput_2s = outputs_2s[j] * mask_2s
                loss_2s += criterion(stackoutput_2s, heatmaps_targets_2s)
        loss = loss + loss_2s


        # for j in range(config['nstack'] + 2):
        #     if(j!= config['nstack']-1):
        #         stackoutput = outputs[j] * mask
        #         loss += criterion(stackoutput, heatmaps_targets)
        #     else:
        #         stackoutput = outputs[j] * mask
        #         loss_gm = criterion(stackoutput, heatmaps_targets)
        #     # 2s
        #     if (j < config['nstack']+1):
        #         stackoutput_2s = outputs_2s[j] * mask_2s
        #         loss_2s += criterion(stackoutput_2s, heatmaps_targets_2s)
        # loss =  loss + 1.4 * loss_gm + loss_2s



        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        #gt size
        gts = gts.numpy()
        gts = gts * [ 2 , 2]
        # gts = gts.numpy() * [1 / 2, 1 / 2]


        # evaluate
        all_peak_points = get_peak_points(outputs_2s[config['nstack']].cpu().detach().numpy())
        # all_peak_points = get_peak_points(outputs[config['nstack']].cpu().detach().numpy())
        all_peak_points = all_peak_points * [2, 2]
        rmse = get_mse(all_peak_points, gts, indices_valid_2s)
        train_rmse += rmse

        
        # PCK
        acc,aveacc,_ = pck.eval(all_peak_points, gts, 0.2)
        train_PCK += aveacc

        #  each part pck
        for k in range(0, 5):
            if acc[k] != -1:
                pckcount[k] += 1
                train_PCK2[k] += acc[k]
        print('[ train---Epoch {:005d} -> {:005d} / {} ] loss : {:15} rmse : {:15} allPCK : {:5} {:5} {:5} {:5} {:5}'.format(
            epoch, i * config['batch_size'], config['train_num'], loss, rmse, acc[0], acc[1], acc[2], acc[3], acc[4]))


    trainallPCK = np.true_divide(train_PCK2, pckcount)
    AvetrainPCK = (trainallPCK[1] + trainallPCK[2] + trainallPCK[3] + trainallPCK[4]) / 4
    train_loss_everyepoch = train_loss / (i + 1)
    train_rmse_everyepoch = train_rmse / (i + 1)
    train_PCK_everyepoch = train_PCK / (i + 1)


    # save result
    dict = {'epoch': 0, 'lr': 0, 'loss': 0, 'rmse': 0, 'avePCKepoch': 0 , 'avePCK' : 0, 'part1PCK' : 0,
            'part2PCK' :0,  'part3PCK' :0, 'part4PCK' :0}
    dict['epoch'] = epoch
    dict['lr'] = config['lr']
    dict['loss'] = train_loss_everyepoch
    dict['rmse'] = train_rmse_everyepoch.numpy()
    dict['avePCKepoch'] = train_PCK_everyepoch
    dict['avePCK'] = AvetrainPCK
    dict['part1PCK'] = trainallPCK[1]
    dict['part2PCK'] = trainallPCK[2]
    dict['part3PCK'] = trainallPCK[3]
    dict['part4PCK'] = trainallPCK[4]

    df = pd.DataFrame([dict])
    if epoch == 0:
        df.to_csv(trainresultname)
    else:
        df.to_csv(trainresultname, mode='a', header=False)
    print('[ train---Epoch {:005d} ] loss : {:15} rmse : {:15} allPCK : {:5} {:5} {:5} {:5} {:5} {:5} '.format(epoch, train_loss_everyepoch, train_rmse_everyepoch, train_PCK_everyepoch,
                                                                                  AvetrainPCK, trainallPCK[1], trainallPCK[2],trainallPCK[3], trainallPCK[4]))

    return net, (train_loss_everyepoch,train_rmse_everyepoch, train_PCK_everyepoch)


def valid(net, validDataLoader,epoch):
    lowrmse = 100000
    valid_loss = 0
    valid_rmse = 0
    valid_PCK = 0
    valid_PCK2 = [0, 0, 0, 0, 0]
    pckcount = [0, 0, 0, 0, 0]
    net.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(validDataLoader):
            inputs = data['image']
            heatmaps_targets = data['heatmaps']
            heatmaps_targets_2s = data['heatmaps_2s']
            gts = data['keypoints']
            mask, indices_valid, indices_notvalid = calculate_mask(heatmaps_targets)
            heatmaps_targets = heatmaps_targets * mask

            #2s
            mask_2s, indices_valid_2s, indices_notvalid_2s = calculate_mask(heatmaps_targets_2s)
            heatmaps_targets_2s = heatmaps_targets_2s * mask_2s


            inputs = inputs.to(device)
            heatmaps_targets = heatmaps_targets.to(device)
            mask = mask.float().to(device)

            #2s
            heatmaps_targets_2s = heatmaps_targets_2s.to(device)
            mask_2s = mask_2s.float().to(device)

            outputs, outputs_2s = net(inputs)

            # intermideate supervision
            loss = 0
            loss_2s = 0
        
           
            for j in range(config['nstack']+2):
                stackoutput = outputs[j] * mask  # outputs为8次stack每次输出预测值，8哥tensor的列表
                loss += criterion(stackoutput, heatmaps_targets)
                # 2s
                if (j < config['nstack']+1 ):
                    stackoutput_2s = outputs_2s[j] * mask_2s  # outputs为8次stack每次输出预测值，8哥tensor的列表
                    loss_2s += criterion(stackoutput_2s, heatmaps_targets_2s)
            loss = loss + loss_2s
            valid_loss += loss.item()

            # gt size
            gts = gts.numpy()
            gts = gts* [ 2 , 2]
            # gts = gts.numpy() * [1 / 2, 1 / 2]

            all_peak_points = get_peak_points(outputs_2s[config['nstack']].cpu().detach().numpy())
            # all_peak_points = get_peak_points(outputs[config['nstack']].cpu().detach().numpy())
            all_peak_points = all_peak_points * [2, 2]
            rmse = get_mse(all_peak_points, gts, indices_valid_2s)  # 关键点坐标的loss
            valid_rmse += rmse

            # PCK
            acc, aveacc, _ = pck.eval(all_peak_points, gts, 0.2)
            valid_PCK += aveacc

            # 计算每个部位的pck
            for k in range(0, 5):
                if acc[k] != -1:
                    pckcount[k] += 1
                    valid_PCK2[k] += acc[k]
            print('[ valid---Epoch {:005d} -> {:005d} / {} ] loss : {:15} rmse : {:15} allPCK : {:5} {:5} {:5} {:5} {:5} {:5}'.format(
                epoch, i, config['valid_num'], loss, rmse, aveacc, acc[0], acc[1], acc[2], acc[3], acc[4]))

        validallPCK = np.true_divide(valid_PCK2, pckcount)
        AvevalidPCK = (validallPCK[1] + validallPCK[2] + validallPCK[3] + validallPCK[4]) / 4
        
        valid_loss_everyepoch = valid_loss / (i + 1)
        valid_rmse_everyepoch = valid_rmse / (i + 1)
        valid_PCK_everyepoch = valid_PCK / (i + 1)

        # save result
        validdict = {'epoch': 0, 'lr': 0,'loss': 0, 'rmse': 0, 'avePCKepoch': 0, 'avePCK': 0, 'part1PCK': 0,
                'part2PCK': 0, 'part3PCK': 0, 'part4PCK': 0, }
        validdict['epoch'] = epoch
        validdict['r'] = config['lr']
        validdict['loss'] = valid_loss_everyepoch
        validdict['rmse'] = valid_rmse_everyepoch.numpy()
        validdict['avePCKepoch'] = valid_PCK_everyepoch
        validdict['avePCK'] = AvevalidPCK
        validdict['part1PCK'] = validallPCK[1]
        validdict['part2PCK'] = validallPCK[2]
        validdict['part3PCK'] = validallPCK[3]
        validdict['part4PCK'] = validallPCK[4]

        df = pd.DataFrame([validdict])
        if epoch == 0:
            df.to_csv(validresultname)
        else:
            df.to_csv(validresultname, mode='a', header=False)

        print('[ valid---Epoch {:005d} ] loss : {:15} rmse : {:15} allPCK : {:5}, {:5} {:5} {:5} {:5} {:5}'.format( epoch, valid_loss_everyepoch, valid_rmse_everyepoch, valid_PCK_everyepoch,
                                                                                       AvevalidPCK, validallPCK[1],
                                                                                       validallPCK[2], validallPCK[3],
                                                                                       validallPCK[4]
                                                                                       ))

    # Save checkpoint.
    saveepochcheckpont_PCK(valid_PCK_everyepoch, config['highPCK'], epoch, net, netname)
    return (valid_loss_everyepoch,valid_rmse_everyepoch, valid_PCK_everyepoch)





if __name__ == '__main__':
    pck = PCK(njoints=4)
    writer = SummaryWriter(comment='mousepose_train')
    writer_total = SummaryWriter(comment='mousepose_valid')
    net,trainDataLoader, validDataLoader = initnet()
    updatelr = []
    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        running_loss = 0.0
        nettrain, (train_loss_everyepoch,train_rmse_everyepoch, train_PCK_everyepoch) = train(net, trainDataLoader, epoch)
        (valid_loss_everyepoch,valid_rmse_everyepoch, valid_PCK_everyepoch) = valid(nettrain, validDataLoader, epoch)
        updatelr.append(valid_rmse_everyepoch)
        writer.add_scalar('loss', train_loss_everyepoch, global_step=epoch)
        writer.add_scalar('RMSE', train_rmse_everyepoch, global_step=epoch)
        writer.add_scalar('PCK', train_PCK_everyepoch, global_step=epoch)

        writer_total.add_scalar('loss', valid_loss_everyepoch, global_step=epoch)
        writer_total.add_scalar('RMSE', valid_rmse_everyepoch, global_step=epoch)
        writer_total.add_scalar('PCK', valid_PCK_everyepoch, global_step=epoch)

