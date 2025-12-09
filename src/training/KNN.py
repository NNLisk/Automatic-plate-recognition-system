import sys, os
import cv2
import numpy as np


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)
print(project_root)

_ = os.path.join("data", "inference", "sessions")



def load_images(containerPath, dimension=(100,75)):
    image_directory = containerPath
    print(image_directory)
    folders = [d for d in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, d))]
    
    folders.sort()
    print(folders)

    images = []
    labels = []

    description = "An OCR dataset for k-NN algorithm"

    # preprocessing
    for i, d in enumerate(folders):
        fpath = os.path.join(image_directory, d)
        print(d)
        print(fpath)
        for file in os.listdir(fpath):
            print(file)
            file_path = os.path.join(fpath, file)
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) /255.0
            images.append(img)
            labels.append(i)

    x = np.array(images)
    y = np.array(labels)
    return x, y

if __name__ == "__main__":
    x, y = load_images(os.path.join("data", "OCR_training_data", "data", "train"))
    print(y[8990:9000])