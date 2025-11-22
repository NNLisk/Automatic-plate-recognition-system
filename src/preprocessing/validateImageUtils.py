import cv2
import os

# checks images outside 640x640
# apparently yolo can normalize so its fine
def check_image_sizes(folderpath):

    for img_name in os.listdir(folderpath):
        img_path = os.path.join(folderpath, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            h, w = img.shape[:2]
            if h != 640 or w != 640:
                print(img_name, h, w)

check_image_sizes("/home/niko/Programming/school/AI/project/data/Raw/images")

