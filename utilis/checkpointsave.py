import torch
import os
from utilis.netparameter320 import config


def savecheckpoint(epoch, net, netname):


    if (epoch + 1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/' + netname):
            os.mkdir('checkpoint/' + netname)
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/' + netname + '/epoch_{}checkpoint.ckpt'.format(epoch))

 # Save checkpoint.
def saveepochcheckpont(meanrmse, lowrmse, epoch, net, netname):
    if meanrmse < lowrmse:
        print('Saving checkpoint...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/' + netname):
            os.mkdir('checkpoint/' + netname)
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'rmse' : meanrmse
        }
        torch.save(state, './checkpoint/' + netname + '/epoch_{}checkpoint.ckpt'.format(epoch))
        lowrmse = meanrmse
        config['lowrmse'] = meanrmse

def saveepochcheckpont_PCK(meanPCK, highPCK, epoch, net, netname):
    if meanPCK > highPCK:
        print('Saving checkpoint...')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.isdir('checkpoint/' + netname):
            os.mkdir('checkpoint/' + netname)
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'rmse' : meanPCK
        }
        torch.save(state, './checkpoint/' + netname + '/epoch_{}checkpoint.ckpt'.format(epoch))
        highPCK = meanPCK
        config['lowrmse'] = meanPCK