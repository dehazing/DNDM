import torch
import torch.nn as nn
import math
from torch import cat

class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 5, stride, 2, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out
class _ResBlockSR(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBlockSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,5, stride,2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        '''
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()
        '''

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _DeblurringMoudle(nn.Module):
    def __init__(self):
        super(_DeblurringMoudle, self).__init__()
        self.conv1     = nn.Conv2d(3, 16, (5, 5), 1, padding=2)
        self.relu      = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock1 = self._makelayers(16, 16, 6)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3),1, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock2 = self._makelayers(16,16, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )
        self.resBlock3 = self._makelayers(64, 64, 6)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), 1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (4, 4), 1, padding=1),#nn.ConvTranspose2d(128, 64, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (7, 7), 1, padding=2)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, (3, 3), 1, 1)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    #'''
    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)
    #'''
    def forward(self, x):
        con1   = self.relu(self.conv1(x))
        res1   = self.resBlock1(con1)
        res1   = torch.add(res1, con1)
        con2   = self.conv2(res1)
        res2   = self.resBlock2(con2)
        res2   = torch.add(res2, con2)
        con3   = self.conv3(res2)
        res3   = self.resBlock3(con3)
        res3   = torch.add(res3, con3)
        decon1 = self.deconv1(res3)
        deblur_feature = self.deconv2(decon1)
        deblur_out = self.convout(torch.add(deblur_feature, con1))
        return deblur_feature, deblur_out#res3 ,con3#

class _Content1(nn.Module):
    def __init__(self):
        super(_Content1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), 1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock = self._makelayers(16, 16, 8, 1)

        self.conv2 = nn.Conv2d(16, 16, (3, 3), 1, 1)
        '''
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()
        '''

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        res8 = self.resBlock(con1)
        con2 = self.conv2(res8)
        sr_feature = torch.add(con2, res8)  # +x
        Content = cat([con1, res8, con2,sr_feature], 1)
        return sr_feature, Content


class _SRMoudle1(nn.Module):
    def __init__(self):
        super(_SRMoudle1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), 1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resBlock = self._makelayers(16, 16, 8, 1)

        self.conv2 = nn.Conv2d(16, 16, (3, 3), 1, 1)
        '''
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()
        '''


    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBlockSR(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))

        res8 = self.resBlock(con1)
        con2 = self.conv2(res8)
        sr_feature = torch.add(con2, res8)  # +x
        mask = cat([con1,  res8, con2,sr_feature ], 1)
        return sr_feature, mask


class _ReconstructMoudle(nn.Module):
    def __init__(self):
        super(_ReconstructMoudle, self).__init__()
        self.resBlock = self._makelayers(16, 16, 8)
        self.conv1 = nn.Conv2d(16, 16, (3, 3), 1, 1)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(16,32, (3, 3), 1, 1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), 1, 1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(16, 3, (3, 3), 1, 1)
        '''
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()
        '''
    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        res1 = self.resBlock(x)
        con1 = self.conv1(res1)
        con2 = self.conv2(con1)
        con3 = self.relu3(self.conv3(con2))
        sr_deblur = self.conv4(con3)
        return sr_deblur

class Net_content(nn.Module):
    def __init__(self):
        super(Net_content, self).__init__()
        self.content_Moudle  = self._make_net(_Content1)

    def forward(self, x):

        content_out  = self.content_Moudle(x)

        return content_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)


class Net_hazy(nn.Module):
    def __init__(self):
        super(Net_hazy, self).__init__()
        self.hazy_Moudle  = self._make_net(_SRMoudle1)

    def forward(self, x):

        hazy_out = self.hazy_Moudle(x)

        return hazy_out#, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)

class Net_mask(nn.Module):
    def __init__(self):
        super(Net_mask, self).__init__()
        self.hazyMoudle = self._make_net(_DeblurringMoudle)

    def forward(self, x):

        mask_out, mask_feature = self.hazyMoudle(x)


        return mask_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)

class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, content, hazy_mask):


        repair_feature = torch.mul(content, hazy_mask)
        fusion_feature = torch.add(content, repair_feature)
        recon_out = self.reconstructMoudle(fusion_feature)


        return recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)


