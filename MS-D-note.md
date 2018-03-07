

## Mixed-Scale Dense NN (MS-D)
A *very deep* network with *few channels* in each layer. 

Key idea: Mixed-scale dilated convolution + Dense connection

### Mixed-scale dilated convolution 

By formula (3), it applies convolution with different scale of [dilations](https://arxiv.org/pdf/1511.07122.pdf) to different channels of the layer and sum the results.

In TensorFlow/Caffe/PyTorch, convolution function sets parameters `input channels, output channels, dilation` for the whole layer in a single step, making convolutions over all channels have the same dilation. Theoratically speaking, one can still unpack a multi-channel feature map into multiple one-channel feature maps, and then apply differently dilated convolutions over them and sum the results. But the code can be long and messy. Efficiency might be a bigger problem, one framework buit-in matrix operation is decomposed into n seperate operations, which might be slow. This is probably the reason the paper uses PyCUDA to implement its own operator.

Advantages:

- Information of all scales (especially large scales) are available in early stages of the network. Can have useful information to improve deeper layers (better for optimizing globally).

- With information of multiple scales all being considered within one single operation, it's more convenient for network to optimize (*inform decisions at other scales* even within one layer), which potentially reduces paramters (eg. avoid downscaling/upscaling).

	(advantage 1&2 has some sort of overlap?)

- Network learns a combination of dilations to perform well in different problems with different conditions. Analogical to the idea of parallely using different kernel sizes in an inception block in [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf). The network learns to use the better kernel size in a specific layer with inception blocks.

![inception block](https://github.com/hardyqr/DL_for_predicting_protein_contact_map/blob/master/inception.png)

### Dense connections

Premise: with mixed-scale dilated convolution, we don't use downsampling or upsampling, which means all feature maps have identical sizes (length and width).

=> Idea of [DenseNet](https://arxiv.org/pdf/1608.06993.pdf): can use all previous layers for creating a new layer, adding as more inter-layer connection as possible (if L layers in total, there would be L×(L+1)/2 connections). By formula (4), like DenseNet, every new layer is obtained by summing convolutions over all previous layers (feature maps).

Advantages:

- Feature maps are maximally used.
	- Encourage reusing feature maps: some early layer information may be useful for later layers, won't have to reproduce them => reduce parameters.
	- Similar to ResNet, feature maps can be either used or not used, depending on which yields a better result. So, the network would have at least the performance of not having all previous feature maps.

- Prevents vanishing-gradient: all layers have direct access to the gradients of the loss

In experiment, the paper only used one channel. But according to formula `sij=((i*w+j) mod 10)+1`, if `w=1`,`j=1`, then at all times there is `sij = (i+1) mod 10 + 1`, the receptive field of filter would grow much slower. The maximum receptive field won't be reached until 8th layer. And it (area of receptive field) goes back and forth. Within one layer, since there is only one channel, there can only be one kind of dilated convolution. This leads to less variation of receptive fields within one layer (Isn't this making some of the claimed mixed-scale advantages invalid?). This results the network to be kind of similar to DenseNet but using periodic dilations. So, *reduction of parameters* might have mainly benefited from the Dense block mechanism.

And from another perspective, if all layers are densely connected, how much difference there is to have only different dilations in different layers comparing to mixed dilations within one layer?
