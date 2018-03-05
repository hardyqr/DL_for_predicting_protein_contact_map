# March 5th ,2018 @DP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from res_layers import *
from utils import *


class Unet(nn.Module):
    def __init__(self, channels_in=26, classes_out=1):
        super(Unet, self).__init__()
        # inconv, outconv
        self.inconv = oned_res_conv(channels_in,32)
        self.outconv = outconv(256,classes_out)
        self.logistic_regression = torch.nn.Linear(classes_out,classes_out, bias=False)

        # 1d network
        self.oned_res_32 = oned_res_conv(32,32)
        self.oned_32_to_64 = oned_res_conv(32,64)
        self.oned_res_64 = oned_res_conv(64,64)
        self.oned_64_to_128 = oned_res_conv(64,128)
        self.oned_res_128 = oned_res_conv(128,128)

        # 2d network
        self.twod_res_128 = twod_res_conv3x3(128,128)
        self.twod_128_to_256 = twod_res_conv3x3(128,256)
        self.twod_res_256 = twod_res_conv3x3(256,256)

    def forward(self, x):
        print("L: ",x.shape[-1])
        print("********** Network 1 begins ********** ")
        x = self.inconv(x) 

        # residual block 1, 32x32
        res = self.oned_res_32(x) 
        #p16d = (1, 31) # pad last dim by 16 on each side
        #res = F.pad(res,p16d,'constant',0)
        x = x + res
        #print('after block 1: ', x.shape)

        # residual block 2, 32x32
        res = self.oned_res_32(x) 
        x = x + res
        #print('after block 2: ', x.shape)
        
        # residual block 3, 32->64, add new channels (features)
        x = self.oned_32_to_64(x)
        #print('after block 3: ', x.shape)

        # residual block 4, 64x64
        res = self.oned_res_64(x) 
        x = x + res
        #print('after block 4: ', x.shape)

        # residual block 5, 64->128, add new channels 
        x = self.oned_64_to_128(x)
        #print('after block 5: ', x.shape)

        # residual block 6, 128x128
        res = self.oned_res_128(x) 
        x = x + res
        #print('after block 6: ', x.shape)

        # sequence -> pairwise matrix
        t = x.data.numpy().transpose([0,2,1])
        #print('sequence shape: ',t.shape)
        matrix_t = seq2pairwise(t)
        #print('pairwise matrix shape: ', matrix_t.shape)

        # stack coevolution info, pairwise potential
        ba, L,_,_ = matrix_t.shape
        green = np.random.rand(ba,L,L,3)
        #print('green matrix shape: ',green.shape)

        n2input = np.concatenate((matrix_t, green),axis = -1)
        print('concatenated batch_size x L x L x (3n+3) shape: ')
        print( n2input.shape)
        print("********** Network 2 begins ********** ")
        
        x = Variable(torch.FloatTensor(n2input.transpose([0,3,1,2])))

        # change channel size: 3+3n -> 128
        self.chto128= twod_res_conv3x3(x.shape[1],128)
        x = self.chto128(x)
        #print('after 2d block 0: ', x.shape)

        # 2dresidual block 1, 128x128
        res = self.twod_res_128(x) 
        x = x + res
        #print('after 2d block 1: ', x.shape)
         
        # 2dresidual block 2, 128x128
        res = self.twod_res_128(x) 
        x = x + res
        #print('after 2d block 2: ', x.shape)       
        
        # 2dresidual block 3, 128->256
        x = self.twod_128_to_256(x)

        #print('after 2d block 3: ', x.shape)

        # 2dresidual block 4, 256x256
        res = self.twod_res_256(x) 
        x = x + res
        #print('after 2d block 4x: ', x.shape)

        # 2dresidual block 5, 256x256
        res = self.twod_res_256(x) 
        x = x + res
        #print('after 2d block 5: ', x.shape)


        # outconv - 1x1 conv
        x = self.outconv(x)
        # We don't need the softmax layer here since in PyTorch, 
        # CrossEntropyLoss already uses it internally


        return x

if __name__ == "__main__":
    """
    testing
    """
    model = Unet(26,1) # input channel, output channel
    for _ in range(0,5):
        """
        input:
            batch_size - random: 5 - 10
            channel_size - 26 (fixed)
            sequence_size - random: 150 - 200
        """
        print('---------- batch ' + str(_+1) + ' ----------')
        input = np.random.random((np.random.randint(5,11),26,np.random.randint(150,200)))
        #print(input.shape)
        x = Variable(torch.FloatTensor(input))
        out = model(x)
        np_out = out.data.numpy().transpose([0,2,3,1])
        print('Final output shape batch_size x L x L x output_channel: ')
        print(np_out.shape)
        loss = torch.sum(out)
        loss.backward()
