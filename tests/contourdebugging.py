import cv2
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import thresholded_2_segmented_letters

def show_contours_debug():
    session = "data/inference/sessions/4/"
    rects = thresholded_2_segmented_letters(
        f"{session}thresholded.jpg",session
    )

    img = cv2.imread(f"{session}cropped.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(rects):
        print(x,y, w, h)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, str(i), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    scaled = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{session}contours.jpg", scaled)
    cv2.imshow("Contour Debug", scaled)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    show_contours_debug()