# March 5 ,2018

import torch
import torch.nn as nn
import torch.nn.functional as F


def oned_res_conv(in_ch,out_ch):
    conv_seq = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv1d(in_ch,out_ch,17,padding=8),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch,17,padding=8),
                )
    return conv_seq

def twod_res_conv3x3(in_ch,out_ch):
    conv_seq = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch,out_ch,3,padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch,3,padding= 1),
                )
    return conv_seq

def twod_res_conv5x5(in_ch,out_ch):
    conv_seq = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch,out_ch,5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch,5,padding=2),
                )
    return conv_seq

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv,self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,1) # 1x1 conv
    def forward(self, x):
        x = self.conv(x)
        return x
