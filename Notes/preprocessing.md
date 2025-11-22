# Preprocessing notes

## IMAGE PREPROCESSING FOR TRAINING YOLOv8

+ Images are grayscaled
+ For data augmentation, 5% salt and pepper noise is added
    + this makes the plate finding model more robust
    + YOLO is optimized for real world dirty lenses, messy images and other imperfections
    + no unified sizing needed for the plate finder since yolo can normalize images to its standard input size 640x640