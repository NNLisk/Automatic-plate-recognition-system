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

+ v1: just training, no validation no augmentation
+ v2: added validation to save the best model and avoid overfitting
+ v3: test script to provide metrics and added data augmentation to overcome real world image imperfections. Meaningful improvements
    + Still issues with the letters I, A, B
+ v4: trained for 100 epochs instead of 40 -> no meaningful improvement
+ v5: added another convolutional layer

+ IMPROVEMENT IDEAS:
    + add another convolutional layer, or a fully connected layer
    + more augmentation?

    
## refs

+ CNN tutorials followed
    + https://www.digitalocean.com/community/tutorials/writing-cnns-from-scratch-in-pytorch
    + https://medium.com/latinxinai/convolutional-neural-network-from-scratch-6b1c856e1c07
    + https://www.datacamp.com/tutorial/pytorch-cnn-tutorial (main one)

+ torchmetrics
    + https://www.evidentlyai.com/classification-metrics/multi-class-metrics