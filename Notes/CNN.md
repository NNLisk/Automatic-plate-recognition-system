# Notes on Convolutional neural networks

+ initially planned structure
    + 3 convolutional layers
    + maxpooling layer
    + flattening
    + one fully connected layer

+ first predictions: 
    + predicted: I, when actually was: A
    + predicted: E, when actually was: E
    + predicted: 0, when actually was: 0
    + predicted: Z, when actually was: 2
    + predicted: A, when actually was: I
    + predicted: Y, when actually was: Y
    + predicted: B, when actually was: B
    + predicted: 6, when actually was: 6
    + predicted: 7, when actually was: 7
    + predicted: 0, when actually was: 0
    + predicted: 5, when actually was: 5

    + 64% success rate


## refs

+ CNN tutorials followed
    + https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
    + https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07
    + https://www.datacamp.com/tutorial/pytorch-cnn-tutorial (main one)