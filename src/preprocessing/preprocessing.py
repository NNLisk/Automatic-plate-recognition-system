import cv2


# preprocessing method ran on each picture coming from the UI

def preprocessForPlateFinding(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

