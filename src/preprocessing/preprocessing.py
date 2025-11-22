import cv2

def preprocess(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
