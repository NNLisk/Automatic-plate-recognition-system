# Training related 

## YOLOv8 nano
+ trained for finding the plates from the images
+ explanation:
    + model trains on raw images defined in preprocessing.md it uses annotations to check and improve its estimates it checks each batch 50 times (epochs)
    + for now, batches of 16 for 50 epochs
    + (362 / 16) x 50 = 1131 updates to weights roughly
    + training file: src/training/tainYOLOv8.py
+ training results
    + Confidence graphs at models/plate_detector
    + MOST IMPORTANT ONE:
        + results.jpg -> shows that training loss goes down, normal, but also validation loss devreases too
        + logarithmic decrease is good, MLA learns fast
        + confusion matrix: 98% of plates detected, 2% false negatives
        + YOLO doesnt mispredict or isnt overfit
        + YOLO IS GOOD TO DETECT PLATES FROM IMAGES

+ References
    + https://docs.ultralytics.com/modes/train/#idle-gpu-training
