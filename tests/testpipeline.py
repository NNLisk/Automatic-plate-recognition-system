import cv2
import os
from ultralytics import YOLO
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.pipeline import processImage
from src.utils.filer import make_new_session


if __name__ == "__main__":

    sp, sid = make_new_session()
    imagefile = "test2.jpg"

    img = cv2.imread(os.path.join("data", "inference", "raw", imagefile))
    print(sp, sid)
    cv2.imwrite(os.path.join(sp, "rawinput.jpg"), img)

    # result_data, plate, confidence_values = processImage(sp, sid, "CNN")
    # print(plate)

    plateKNN = processImage(sp, sid, "KNN")
    print("plate with KNN found: " + plateKNN)