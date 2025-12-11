import cv2
import os, sys


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)


fpath = os.path.join("data", "inference", "sessions", "1", "characters", "2.jpg")
print(fpath)

img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("s", img)
cv2.waitKey(0)

img = cv2.resize(src=img, fx=0.1, dsize=(100,75))

cv2.imshow("s", img)
cv2.waitKey(0)