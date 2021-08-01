import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from model.keypointspooling import LeftPool, TopPool, BottomPool,RightPool

class ConvBnRelu(nn.Module):
    """docstring for BnReluConv"""
    def __init__(self, inChannels, outChannels, kernelSize=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
        self.bn = nn.BatchNorm2d(self.outChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BnReluConv(nn.Module):
        """docstring for BnReluConv"""
        def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
                super(BnReluConv, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.kernelSize = kernelSize
                self.stride = stride
                self.padding = padding

                self.bn = nn.BatchNorm2d(self.inChannels)
                self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
                self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
                x = self.bn(x)
                x = self.relu(x)
                x = self.conv(x)
                return x


class ConvBlock(nn.Module):
        """docstring for ConvBlock"""
        def __init__(self, inChannels, outChannels):
                super(ConvBlock, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.outChannelsby2 = outChannels//2

                self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
                self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
                self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

        def forward(self, x):
                x = self.cbr1(x)
                x = self.cbr2(x)
                x = self.cbr3(x)
                return x

class SkipLayer(nn.Module):
        """docstring for SkipLayer"""
        def __init__(self, inChannels, outChannels):
                super(SkipLayer, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                if (self.inChannels == self.outChannels):
                        self.conv = None
                else:
                        self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

        def forward(self, x):
                if self.conv is not None:
                        x = self.conv(x)
                return x

class Residual(nn.Module):
        """docstring for Residual"""
        def __init__(self, inChannels, outChannels):
                super(Residual, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.cb = ConvBlock(inChannels, outChannels)
                self.skip = SkipLayer(inChannels, outChannels)

        def forward(self, x):
                out = 0
                out = out + self.cb(x)
                out = out + self.skip(x)
                return out


class FeaEnhance(nn.Module):
        """docstring for Residual"""
        def __init__(self, inChannels, outChannels):
                super(FeaEnhance, self).__init__()
                super(FeaEnhance, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.outChannelsby2 = outChannels // 2

                # s1
                self.atrousconv1 = nn.Conv2d(self.inChannels, self.outChannelsby2, 3, stride=2, dilation=2, padding=2)
                self.maxp1 = nn.MaxPool2d(2, 2)
                self.conv1 = nn.Conv2d(self.inChannels, self.outChannelsby2, 1, 1, 0)
                self.bnrc1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)

                # s2
                self.maxp2 = nn.MaxPool2d(2, 2)
                self.upsampling1 = nn.Upsample(scale_factor=2)
                self.maxp3 = nn.MaxPool2d(2, 2)
                self.upsampling2 = nn.Upsample(scale_factor=2)

                self.bnrc2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
                self.atrousconv2 = nn.Conv2d(self.outChannelsby2, self.outChannelsby2, 3, stride=4, dilation=2,
                                             padding=2)
                self.maxp4 = nn.MaxPool2d(4, 4)

                # s3
                self.upsampling4 = nn.Upsample(scale_factor=4)
                self.conv2 = nn.Conv2d(self.outChannelsby2, self.outChannels, 1, 1, 0)
                self.upsampling5 = nn.Upsample(scale_factor=4)
                self.conv3 = nn.Conv2d(self.outChannelsby2, self.outChannels, 1, 1, 0)

                self.bnrc3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

                self.skip = SkipLayer(inChannels, outChannels)

        def forward(self, x):
            # s1
            out1 = self.atrousconv1(x)  # 240*320*128
            out2 = self.maxp1(x)  # 120*160*128
            out2 = self.conv1(out2)
            out3 = self.bnrc1(x)

            # s2
            out1s2_1 = self.maxp2(out1)
            out1s2_2 = self.upsampling1(out1)

            out2s2_1 = self.maxp3(out2)
            out2s2_2 = self.upsampling2(out2)

            out3s2_1 = self.atrousconv2(out3)
            out3s2_2 = self.maxp4(out3)
            out3s2_3 = self.bnrc2(out3)

            out1s2 = out1s2_1 + out3s2_1
            out2s2 = out2s2_1 + out3s2_2
            out3s2 = out3s2_3 + out1s2_2 + out2s2_2

            # s3
            out1s3 = self.upsampling4(out1s2)
            out1s3 = self.conv2(out1s3)

            out2s3 = self.upsampling5(out2s2)
            out2s3 = self.conv3(out2s3)

            out3s3 = self.bnrc3(out3s2)

            out = out1s3 + out3s3 + out2s3
            out = out + self.skip(x)
            return out



class Feaenhance_2(nn.Module):
        """docstring for Feaenhance"""
        def __init__(self, inChannels, outChannels):
                super(Feaenhance_2, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                # self.outChannelsby2 = outChannels // 2

                #s1
                self.bnrc1 = BnReluConv(self.inChannels, self.outChannels, 1, 1, 0)

                #s2
                self.bnrc2 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.maxp1 = nn.MaxPool2d(2, 2)
                self.maxp2 = nn.MaxPool2d(4, 4)

                #s3
                self.bnrc3 = BnReluConv(self.outChannels, self.outChannels, 1, 1, 0)
                self.maxp3 = nn.MaxPool2d(2, 2)
                self.maxp4 = nn.MaxPool2d(4, 4)

                self.bnrc4 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.conv1 = nn.Conv2d(self.outChannels, self.outChannels, 1, 1, 0)
                self.upsampling1 = nn.Upsample(scale_factor=2)
                self.maxp5 = nn.MaxPool2d(2, 2)
              


                self.bnrc5 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.conv2 =  nn.Conv2d(self.outChannels, self.outChannels, 1, 1, 0)
                self.upsampling2= nn.Upsample(scale_factor=4)
                self.upsampling3 = nn.Upsample(scale_factor=2)

                self.upsampling4 = nn.Upsample(scale_factor=2)
                self.upsampling5 = nn.Upsample(scale_factor=4)

                self.conv3 =  nn.Conv2d(self.outChannels * 3, self.outChannels, 1, 1, 0)


        def forward(self, x):
            out1 = self.bnrc1(x)

            out2 = self.bnrc2(out1)
            out2_1 = self.maxp1(out1)
            out2_2 = self.maxp2(out1)
            # out2_1 = F.max_pool2d(out1, (2,2))
            # out2_2 =  F.max_pool2d(out1, (4,4))


            out3 = self.bnrc3(out2)
            out3_1 = self.maxp3(out2)
            out3_2 = self.maxp4(out2)

            out4 = self.bnrc4(out2_1)
            out4_1 = self.upsampling1(self.conv1(out2_1))
            out4_2 = self.maxp5(self.conv1(out2_1))

            out5 = self.bnrc5(out2_2)
            out5_1 = self.upsampling2(self.conv2(out2_2))
            out5_2 = self.upsampling3(self.conv2(out2_2))

            out3 = out3 + out4_1 + out5_1
            out4 = out4 + out3_1 + out5_2
            out5 = out5 + out3_2 + out4_2

            out4 = self.upsampling4(out4)
            out5 = self.upsampling5(out5)

            out = torch.cat((out3, out4), dim=1)
            out = torch.cat((out, out5), dim=1)

            out = self.conv3(out)
            return out




class MeanFieldUpdate(nn.Module):
    """
    Meanfield updating

    """

    def __init__(self, inchannels, outchannels): #512，512，512
        super(MeanFieldUpdate, self).__init__()

        self.attenmap = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1 )# K g
        self.attenmap_2 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1)#K

        self.attenmap_g = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1)#g
        self.attenmap_a = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1)


        self.norm_atten_a = nn.Sigmoid()
        self.norm_atten_g = nn.Sigmoid()


        self.message_f = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1) #g

        self.message_f_2 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1)
        #update y_s
        self.update3 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1) #g
                                
        self.update3_2 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1)
    def forward(self, x_s, x_S):   # s and S scale
        # update attention map
        # a_s = torch.cat((x_s, x_S), dim=1)
        # a_s = self.atten_f(a_s)
        # a_s = self.norm_atten_f(a_s)
        ori_x_S = x_S
        

        ori_x_S = self.attenmap(ori_x_S)
        g_s = ori_x_S.mul(x_s)

        x_S = self.attenmap_2(x_S) 
        a_s_0 = x_S.mul(x_s)

        g_s_1 = self.attenmap_g(a_s_0) #(A,G)
        a_s_1 = self.attenmap_a(g_s) #(A,G)
     
        g_s = self.norm_atten_g(g_s+g_s_1)
        a_s = self.norm_atten_a(a_s_0+a_s_1)


        # update the last scale feature map y_S
        y_s = self.message_f(x_s)
        y_S = y_s.mul(g_s)  

        y_s_a = self.message_f_2(x_s)
        y_S_a = y_s_a.mul(a_s) 

        y_S = x_S + y_S + y_S_a # eltwise sum
        


        #update the y_s
        y_S_1 = self.update3(y_S)
        y_s = y_S_1.mul(g_s)  # production

        y_S_a = self.update3_2(y_S)
        y_s_a = y_S_a.mul(a_s)

        y_s = x_s + y_s + y_s_a # eltwise sum

        return y_S, y_s

class attentionCRF(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(attentionCRF, self).__init__()
        # the first meanfield updating

        self.meanFieldUpdate1_1 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_2 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_3 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_4 = MeanFieldUpdate(inChannels, outChannels)

        # #the second meanfield updating
        # self.meanFieldUpdate2_1 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate2_2 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate2_3 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate2_4 = MeanFieldUpdate(inChannels, outChannels)
        #
        # # the third meanfield updating
        # self.meanFieldUpdate3_1 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_2 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_3 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_4 = MeanFieldUpdate(inChannels, outChannels)
        #
        # # the fourth meanfield updating
        # self.meanFieldUpdate3_1 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_2 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_3 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_4 = MeanFieldUpdate(inChannels, outChannels)
        #
        # # the fifth meanfield updating
        # self.meanFieldUpdate3_1 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_2 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_3 = MeanFieldUpdate(inChannels, outChannels)
        # self.meanFieldUpdate3_4 = MeanFieldUpdate(inChannels, outChannels)


    def forward(self, lpool, tpool, rpool, bpool, p):
        # five meanfield updating

        y_S, y_s_1 = self.meanFieldUpdate1_1(lpool, p)
        y_S, y_s_2 = self.meanFieldUpdate1_2(tpool, y_S)
        y_S, y_s_3 = self.meanFieldUpdate1_3(rpool, y_S)
        y_S, y_s_4 = self.meanFieldUpdate1_4(bpool, y_S)
        #

        for i in range(1):
            y_S, y_s_1 = self.meanFieldUpdate1_1(y_s_1, y_S)
            y_S, y_s_2 = self.meanFieldUpdate1_2(y_s_2, y_S)
            y_S, y_s_3 = self.meanFieldUpdate1_3(y_s_3, y_S)
            y_S, y_s_3 = self.meanFieldUpdate1_4(y_s_4, y_S)


        # y_S, y_s_1 = self.meanFieldUpdate2_1(y_s_1, y_S)
        # y_S, y_s_2 = self.meanFieldUpdate2_2(y_s_2, y_S)
        # y_S, y_s_3 = self.meanFieldUpdate2_3(y_s_3, y_S)
        # y_S, y_s_3 = self.meanFieldUpdate2_4(y_s_4, y_S)
        #
        # y_S, y_s_1 = self.meanFieldUpdate2_1(y_s_1, y_S)
        # y_S, y_s_2 = self.meanFieldUpdate2_2(y_s_2, y_S)
        # y_S, y_s_3 = self.meanFieldUpdate2_3(y_s_3, y_S)
        # y_S, y_s_3 = self.meanFieldUpdate2_4(y_s_4, y_S)

        #
        # y_S = self.meanFieldUpdate4_1(lpool, y_S)
        # y_S = self.meanFieldUpdate4_2(tpool, y_S)
        # y_S = self.meanFieldUpdate4_3(rpool, y_S)
        # y_S = self.meanFieldUpdate4_4(bpool, y_S)
        #
        # y_S = self.meanFieldUpdate5_1(lpool, y_S)
        # y_S = self.meanFieldUpdate5_2(tpool, y_S)
        # y_S = self.meanFieldUpdate5_3(rpool, y_S)
        # y_S = self.meanFieldUpdate5_4(bpool, y_S)

        return y_S


class keypointspool(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(keypointspool, self).__init__()
        self.p1_conv1 = ConvBnRelu(inChannels, outChannels, 1) #3*3con,bn,relu
        self.p2_conv1 = ConvBnRelu(inChannels, outChannels, 1)
        self.p3_conv1 = ConvBnRelu(inChannels, outChannels, 1) #3*3con,bn,relu
        self.p4_conv1 = ConvBnRelu(inChannels, outChannels, 1)


        self.p_conv1 = nn.Conv2d(outChannels, outChannels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(outChannels)

        self.conv1 = nn.Conv2d(outChannels, outChannels, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(outChannels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvBnRelu(outChannels, outChannels, 1)

        self.pool1 = LeftPool()
        self.pool2 = TopPool()
        self.pool3 = RightPool()
        self.pool4 = BottomPool()

        self.atCRF = attentionCRF(inChannels, outChannels)

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 3
        p3_conv1 = self.p3_conv1(x)
        pool3 = self.pool3(p1_conv1)

        # pool 4
        p4_conv1 = self.p4_conv1(x)
        pool4 = self.pool4(p2_conv1)

		# attenionCRF
        confusionfea = self.atCRF(pool1, pool2, pool3, pool4)
        p_conv1 = self.p_conv1(confusionfea)
        p_bn1   = self.p_bn1(p_conv1)

        # pool 1 + pool 2
        # p_conv1 = self.p_conv1(pool1 + pool2)
        # p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2



class SSMixer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SSMixer, self).__init__()
        self.p1_conv1 = ConvBnRelu(inChannels, outChannels, 1) #3*3con,bn,relu
        self.p2_conv1 = ConvBnRelu(inChannels, outChannels, 1)
        self.p3_conv1 = ConvBnRelu(inChannels, outChannels, 1) #3*3con,bn,relu
        self.p4_conv1 = ConvBnRelu(inChannels, outChannels, 1)
        self.p5_conv1 = ConvBnRelu(inChannels, outChannels, 1)

        self.conv2 = ConvBnRelu(inChannels, outChannels, 1)

        self.pool1 = LeftPool()
        self.pool2 = TopPool()
        self.pool3 = RightPool()
        self.pool4 = BottomPool()

        self.atCRF = attentionCRF(inChannels, outChannels)

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 3
        p3_conv1 = self.p3_conv1(x)
        pool3 = self.pool3(p1_conv1)

        # pool 4
        p4_conv1 = self.p4_conv1(x)
        pool4 = self.pool4(p2_conv1)

        p5 = self.p5_conv1(x)

		# attenionCRF
        confusionfea = self.atCRF(pool1, pool2, pool3, pool4,p5)

        conv2 = self.conv2(confusionfea)
        return conv2

        # return conv2, pool1, pool2, pool3, pool4, p5



class myUpsample(nn.Module):
     def __init__(self):
         super(myUpsample, self).__init__()
         pass
     def forward(self, x):
         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)
