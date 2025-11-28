# LOG OF CHANGES/UPDATES

## Niko - 22.11
+ started the preprocessing scripts
+ trained the model to find the plates from the images

## Niko - 23.11
+ Working preprocessing
    + can take an image, and return a cropped plate to a specified save file
    + see src/preprocessing/preprocessing.py and tests/test_detection.py
+ wrote testfiles
+ changed data folder structure: will be changed more later

## Niko - 24.11
+ made the pipeline work from raw image to cropped + temp files in one master function: src/pipeline.py
+ filing session system
    + whenever a new raw input image is processed, it creates a folder with a unique id: 001, 002 etc that contains picture specific stuff
+ test results separated to tests/temp
+ tried different thresholds for plates -> regular 127 seems to be the best without technical tests
+ Found the good morphology for the image segmentation

## Niko - 25.11
+ Finished character segmentation
+ statistical filtering for segmentation, images filtered by relative height to each other also

## Niko -  28.11
+ trying out CCPD database for letter classification
+ made the base for the UI - currently with Streamlit tool
+ started building custom cnn with pytorch keras tools