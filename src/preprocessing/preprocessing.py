import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO("models/plate_detector/weights/best.pt")

# preprocessing method ran on each picture coming from the UI

def preprocessForPlateFinding(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("No image")

    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5,5), 0)

    return blurred



def getCroppedPlate(filename):
    for_cropping = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    for_plate_detection = preprocessForPlateFinding(filename)

    if for_cropping is None or for_plate_detection is None:
        raise Exception("No image")
    
    normalized_for_cropping = cv2.normalize(for_cropping, None, 0, 255, cv2.NORM_MINMAX)
    
    temporary_path = "data/inference/preprocessed/temp.jpg"
    cv2.imwrite(temporary_path, for_plate_detection)

    results = model(temporary_path, conf=0.5)
    result = results[0]

    if len(result.boxes) == 0:
        raise Exception("no plates detected")
        return None

    box = result.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    plate_crop = normalized_for_cropping[y1:y2, x1:x2]

    return plate_crop



# for just local tests
# if __name__ == "__main__":
#    os.chdir("/home/niko/Programming/school/AI/project")
#    test_img = "data/inference/raw/test1.jpg"
#    processed = getCroppedPlate(test_img)
#
#    if processed is not None and processed.size > 0:
#        cv2.imwrite("data/inference/preprocessed/testPreprocessed.jpg", processed)
#        print("Saved cropped plate")
#    else:
#        print("Failed either no detection or empty crop")

    

