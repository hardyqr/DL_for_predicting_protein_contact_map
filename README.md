

### Deep learning model for contact prediction
This is a simple PyTorch reproduction of the dataflow of Neural Network mentioned in [Accurate De Novo Prediction of Protein Contact Map by Ultra-Deep Learning Mode](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005324).

The network consists of Network1 (1-D residual network) + Network2 (2-D residual network). It is a mapping from a random sized 1-D sequence (L x 26) to a 2-D matrix (L x L x 1).


As the printed log suggested below, the network can handle different size of sequence as input. Its ouput size depends on input size.
