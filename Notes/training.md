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
        + YOLOv8 IS GOOD TO DETECT PLATES FROM IMAGES

+ References
    + https://docs.ultralytics.com/modes/train/#idle-gpu-training

## FOR OCR

+ Tesseract PSM 7 or 8
+ CRNN/CNN
+ MLP
+ CART decision tree
+ logistic and softmax regression
+ VGG, ResNet, mobileNet
+ PCA, HOG

+ references
    + https://docs.opencv.org/4.x/d9/d1e/tutorial_dnn_OCR.html
    + https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
    + https://www.geeksforgeeks.org/machine-learning/license-plate-recognition-with-opencv-and-tesseract-ocr/   
    + https://docs.pytorch.org/vision/main/models/vgg.html

    
    

## OCR datasets

+ https://www.kaggle.com/datasets/mohamedgobara/plate-license-recognition-dataset
+ https://figshare.com/articles/dataset/Character_classification_data_for_license_plates/3113449