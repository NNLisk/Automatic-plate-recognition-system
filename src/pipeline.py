import os
import sys
import cv2

#gives project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import getCroppedPlate
from src.utils.filer import makeNewSession


def processImage(filename):

    # sessionpath includes the id, id just for reference
    sessionPath, sessionID = makeNewSession()
    inputFileName = f"data/inference/raw/{filename}"
    img = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception(f"Could not read image: {inputFileName}")
    # saves the raw image
    cv2.imwrite(f"{sessionPath}/rawinput.jpg", img)

    cropped = getCroppedPlate(f"{sessionPath}/rawinput.jpg", sessionPath)
    
    cv2.imwrite(f"{sessionPath}/cropped.jpg", cropped)



if __name__ == "__main__":
    processImage("test1.jpg")