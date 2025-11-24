import cv2
import os
from ultralytics import YOLO
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import preprocessForPlateFinding

model = YOLO("models/plate_detector/weights/best.pt")

def testPlateDetection(imgpath):
    # pipeline
    # raw image => preprocessing => detect the plate

    preprocessed = preprocessForPlateFinding(imgpath)

    # writes temporarily to preprocessed data
    temporary_path = "tests/temp/temp.jpg"
    cv2.imwrite(temporary_path, preprocessed)

    results = model(temporary_path, conf=0.5)
    result = results[0]

    annotated = result.plot()
    cv2.imshow("detection result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    box = result.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    plate_crop = preprocessed[y1:y2, x1:x2]

    cv2.imshow('Cropped Plate', plate_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return plate_crop

if __name__ == "__main__":
    # This just runs the plate detection model for testing
    testfile = "data/inference/raw/test1.jpg"
    plate = testPlateDetection(testfile)

    # saves the image to date/inference/preprocessed
    # DELETE AFTER TESTING

    if plate is not None:
        cv2.imwrite("tests/temp/detected_plate.jpg", plate)
        print("Saved cropped plate to data/Preprocessed/")
