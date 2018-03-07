

## Mixed-Scale Dense NN (MS-D)
A very deep network with few channels in each layer. 

Mixed-scale dilated convolution + Dense connection


### Mixed-scale dilated convolution 

By formula (3), it applies convolution with different scale of dilations to different channels of the layer and sum the results.


In TensorFlow/Caffe, convolution function sets (input channels, output channels, dilation) for the whole layer in a single step, making convolutions over all channels have the same dilation. Theoratically speaking, we can still tear a feature map apart into multiple maps (each with one channel), and then apply differently dilated convolution functions and sum the result. But the code can be long and messy. And one matrix operation is decomposed into n operations, that would be slow. This is probably the reason the paper uses PyCUDA to implement its own operator.

advantage: 
- information of all scales (especially large scales) are available in early stages of the network. Can have useful information to improve deeper layers. (better for optimizing globally)

- with information of multiple scales all considered in one single operation, more convenient for network to optimize (inform decisions at other scales even within one layer), which reduced paramters (eg. avoid upscaling).

(advantage 1&2 has some sort of overlap?)

- network learns a combination of dilations to perform well in different problems with different conditions. Analogical to the idea of parallely using different kernel sizes in a inception block in GoogLeNet. The network learns to use the better kernel size in a specific layer with inception blocks.

### Dense connections

with mixed-scale dilation convolution, we don't use downsampling or upsampling, which means all feature maps have identical sizes (length and width). => can use all previous layers for the new layer.

By formula (4), every new layer is obtained by summing convolutions over all previous layers (and its feature maps).

advantage:
- feature maps are maximally used
	- encourage reusing feature maps: some early layer information may be useful for later layers, won't have to reproduce them => reduce parameters
	- And kind of like ResNet, feature maps can be either used or not used, depending on which yields a better result. So, the network would have at least the performance of not having all previous feature maps.
	- prevents vanishing-gradient 

In experiment, only used one channel. But according to formula sij=((iw+j) mod 10)+1, if w=1,j=1 all the time => sij = (i+1) mod 10 + 1, the receptive field of filter would grow much slower and the max receptive field won't be reached until 8th layer. And it goes back and forth. Within one layer, since there is only one channel, there can only be one kind of dilated convolution. Isn't this makeing some of the claimed advantages invalid? This would be kind of similar to densenet but using periodic dilations.
