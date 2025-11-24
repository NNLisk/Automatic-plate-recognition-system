import cv2
import numpy as np
import os
from ultralytics import YOLO

model = YOLO("models/plate_detector/weights/best.pt")

# preprocessing method ran on each picture coming from the UI

def preprocess_for_plate_finding(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("No image")
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized, (5,5), 0)

    return blurred



def get_cropped_plate(filename, sessionPath):
    for_cropping = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    for_plate_detection = preprocess_for_plate_finding(filename)

    if for_cropping is None or for_plate_detection is None:
        raise Exception("No image")
    
    normalized_for_cropping = cv2.normalize(for_cropping, None, 0, 255, cv2.NORM_MINMAX)
    
    temporary_path = f"{sessionPath}/for_plate_detection.jpg"
    cv2.imwrite(temporary_path, for_plate_detection)

    results = model(temporary_path, conf=0.5)
    result = results[0]

    if len(result.boxes) == 0:
        raise Exception("no plates detected")
        return None

    # below: returns the box coordinates for the plat with the highest confidence
    # then crops the normalized plate with the box coordinates
    box = result.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    plate_crop = normalized_for_cropping[y1:y2, x1:x2]

    cv2.imwrite(f"{sessionPath}/cropped.jpg", plate_crop)

    return plate_crop


def process_cropped(filename, sessionPath):
    cropped = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    _, thresheld = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # adaptive_thresholded_gaus = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2
#)   
    # _, thresheld2 = cv2.threshold(cropped, 75, 255, cv2.THRESH_BINARY)
    # _, thresheld3 = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # adaptive_thresholded_mean = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    # adaptive_thresholded_gaus = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    cv2.imwrite(f"{sessionPath}/thresholded.jpg", thresheld)
    # cv2.imwrite(f"{sessionPath}/thresholded75.jpg", thresheld2)
    # cv2.imwrite(f"{sessionPath}/thresholded0otsu.jpg", thresheld3)
    # cv2.imwrite(f"{sessionPath}/thresholded_adaptive_mean.jpg", adaptive_thresholded_mean)
    # cv2.imwrite(f"{sessionPath}/thresholded_adaptive_gaus.jpg", adaptive_thresholded_gaus)

    return thresheld


def thresholded_2_segmented_letters(filename, sessionPath):
    thimg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thimg, cv2.MORPH_OPEN, kernel)
    # cleaned = cv2.dilate(cleaned, kernel,  iterations=2)

    #cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("t", cleaned)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    character_contours = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        plate_h, plate_w = cleaned.shape
        area = w*h / (plate_w * plate_h)
        aspectRatio = h / float(w)
        if area < 0.01 or area > 0.5:
            continue
        if aspectRatio < 0.5 or aspectRatio > 5:
            continue
        if h / plate_h < 0.4:
            continue
        character_contours.append((x,y,w,h))
    return character_contours



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

    

