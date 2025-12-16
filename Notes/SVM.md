# Support vector machine notes (SVM)

+ Finds th optimal hyper-plane to sepperate data into different classes.
+ The main goal of SVM is to maximize the margin (distance between the plane and the closest samples from each class) between the two classes. The larger the margin the better the model performs on new and unseen data.
    + Closest samples are called support vectors
+ SVM learns a compact model during training, meaning predictions are much faster afterward.
+ If data can’t be separated with a straight line, SVM can use a kernel to map data to a higher-dimensional space where it becomes separable.

# How it works
+ Each character image is resized and flattened into a feature vector.
+ SVM learns decision boundaries between character classes during training, so inference is fast (unlike for example KNN, which compares to all training images each time).

# Pro and con
+ Pro: works well in high-dimensional data, good generalization with margin idea.
+ Con: training can be slow on large datasets, and parameters may need tuning.


# Sources
+ https://medium.com/@kushaldps1996/a-complete-guide-to-support-vector-machines-svms-501e71aec19e
+ https://www.geeksforgeeks.org/machine-learning/support-vector-machine-algorithm/


