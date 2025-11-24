import os
import sys
import cv2

#gives project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import get_cropped_plate, process_cropped, thresholded_2_segmented_letters
from src.utils.filer import make_new_session


def processImage(filename):

    # sessionpath includes the id, id just for reference
    sessionPath, sessionID = make_new_session()
    inputFileName = f"data/inference/raw/{filename}"
    img = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception(f"Could not read image: {inputFileName}")
    # saves the raw image
    cv2.imwrite(f"{sessionPath}/rawinput.jpg", img)

    cropped = get_cropped_plate(f"{sessionPath}/rawinput.jpg", sessionPath)
    thresheld = process_cropped(f"{sessionPath}/cropped.jpg", sessionPath)
    



if __name__ == "__main__":
    processImage("test2.jpg")