# Notes for KNN

## K nearest neighbour
+ calculates euclidean distances between the inferred image and all training images
+ we take the 100 nearest target classes, and predict that the inferred character is the class with the highest occurrence out of those 100
+ we chose euclidean distance over just sum of the differences between the pixel values
+ Euclidean distance:
    + square root of the sum of squared differences
    + np.sqrt(np.sum((infer - train)**2))
    + this metric is much stricter with large pixel differencess so we thought itd be better for our case

+ KNN is easy to implement, no training required since it is just a series of mathematical operations (euclidean difference) and then comparing them on the spot.

## results

+ KNN is very slow due to this comparing on the spot but not too slow to wait out. it is a bit less accurate compared to our convolutional neural network, but still gets characters right often enough depending on the plate resolution.

## sources
+ https://medium.com/swlh/image-classification-with-k-nearest-neighbours-51b3a289280
+ https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/
