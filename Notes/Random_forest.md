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

