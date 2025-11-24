# Preprocessing notes

## IMAGE PREPROCESSING FOR TRAINING YOLOv8
+ overview
    + I trained YOLO to detect plates
    + added preprocessing step to save box coordinates and final preprocessed (cropped) image
    + after thresholding --> OCR

+ Image operations and augmentation related (PLATE RECOGNITION)
    + Images are grayscaled
    + 5% salt and pepper noise is added
    + this makes the plate finding model more robust; YOLO is optimized for real world dirty lenses, messy images and other imperfections
    + no unified sizing needed for the plate finder since yolo can normalize images to its standard input size 640x640
    + THRESHOLDING SOLVED; moved from fixed thresholding to fixed + OTSU
+ MORPHOLOGY
    + Thresholded with Binary inv + otsu
    + took soft contours
    + and limited to:
        + contours more than 40% of the plate height
        + more than 1% area, less than 20%
        + more than 1.3 aspect ratio, less than 7
    + HAS ALOT OF ISSUES WITH LOW RESOLUTION IMAGES

+ data splitting and training related (PLATE RECOGNITION)
    + There are 362 images, with one image having multiple variants with different noise

+ pipeline related
    + CURRENTLY: src/preprocessing/preprocessing.py
        + takes in a filepath of a stored image in data/inference/raw
        + saves a temporary file in data/inference/preprocessing, used for plate detection
        + saves a final preprocessed and cropped plate to the same folder
        + TEMPORARY SETTING, we will need user specific folders eventually (SOLVED)
        + folder contains
            + input image
            + temporary cropped plates
            + segmented characters
            + text file with the resulting plate info
        


## resources

+ For image INFERENCE STEP preprocessing:
    + General object detection with YOLO, our detection model is transfer trained
        + https://www.geeksforgeeks.org/deep-learning/introduction-to-object-detection-using-image-processing/#key-steps-in-image-preprocessing
    + Image box coordinates in the preprocessing step
        + https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        + https://docs.ultralytics.com/modes/predict/#boxes
    + Thresholding
        + https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    + morphology
        + https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
        + https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    
