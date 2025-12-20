# random forest classification notes

# v2 notes
-the confidence values in Random Forest stand for how many trees vote for a class, which typically tops at a conservative ~0,5 when running on large data 
-stats.txt variable meanings in RFC:
precision: of all predictions made as this class, the fraction that were correct (TP / (TP+FP)).
recall: of all true samples of this class, the fraction the model found (TP / (TP+FN)).
f1-score: harmonic mean of precision and recall (2·P·R / (P+R)).
support: number of test samples for that class.
accuracy: overall correct predictions / total samples (0.997 on 2840 test characters).
macro avg: average of precision/recall/F1 across classes, unweighted (treats each class equally).
weighted avg: average weighted by support (accounts for class frequency).

# Why doesn't it work the best in real life situations
Although the RFC achieved high accuracy during validation (ass seen in the metrics), its performance drops when used on real license plates. This is due to the fact that HOG is sensitive to image rotation and noise, which can limit its robustness in situatuions like real licence plate testing. HOG assumes consistent edge contrast, stable lighting, and minimal distortions but thats not the case in real world. Small variations cause large changes in the HOG profile, leading to misclassification.

+ https://www.sciencedirect.com/topics/computer-science/histogram-of-oriented-gradient