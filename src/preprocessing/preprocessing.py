import cv2
import numpy as np
import os

imagesPath = "data/inference/"
# preprocessing method ran on each picture coming from the UI

def preprocessForPlateFinding(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise Exception("No image")

    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5,5), 0)

    cv2.imwrite()

    

