# Preprocessing notes

## IMAGE PREPROCESSING FOR TRAINING YOLOv8

+ Image operations and augmentation related
    + Images are grayscaled
    + 5% salt and pepper noise is added
    + this makes the plate finding model more robust; YOLO is optimized for real world dirty lenses, messy images and other imperfections
    + no unified sizing needed for the plate finder since yolo can normalize images to its standard input size 640x640
+ data splitting and training related
    + There are 362 images, with one image having multiple variants with different noise
    

## resources

+ For image INFERENCE STEP preprocessing:
    + https://www.geeksforgeeks.org/deep-learning/introduction-to-object-detection-using-image-processing/#key-steps-in-image-preprocessing

