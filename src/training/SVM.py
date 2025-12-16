import sys, os
import cv2
import numpy as np
from math import floor
import pickle
from sklearn.svm import SVC


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

filepath = os.path.join("data", "inference", "sessions")

dimension = (100, 75)

class_names = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
               'P','Q','R','S','T','U','V','W','X','Y','Z']

model_path = os.path.join("models", "SVM", "v1", "svm_ocr.pkl")

#Loads OCR training images, resize + normalize, and assign labels per class folder
def load_images(root):
    imgs, labels = [], []
    for idx, cname in enumerate(class_names):
        cpath = os.path.join(root, cname)
        if not os.path.isdir(cpath):
            continue
        print(f"LOADING IMAGES: {cname}")
        for fn in os.listdir(cpath):
            fp = os.path.join(cpath, fn)
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            labels.append(idx)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.int32)

#Converts 2D images to flattened 1D vectors
def flatten_images(imgs):
    n = imgs.shape[0]
    flattened = imgs.reshape((n, -1))
    return flattened


#Trains an SVM model for OCR and saves it to disk
def train_svm(data_root=os.path.join("data", "OCR_training_data", "data", "train")):
    imgs, labels = load_images(data_root)
    X = flatten_images(imgs)
    print("Training SVM...")
    svm = SVC(kernel="linear", C=1.0, probability=True)
    svm.fit(X, labels)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    payload = {
        "model": svm,
        "class_names": class_names,
        "dimension": dimension
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"SVM model saved to {model_path}")

#Loads session character images for inference
def _load_chars_for_inference(session_path):
    chpath = os.path.join(session_path, "characters")
    if not os.path.isdir(chpath):
        raise FileNotFoundError(f"Characters folder not found: {chpath}")

    files = sorted(f for f in os.listdir(chpath)
                   if os.path.isfile(os.path.join(chpath, f)))

    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(chpath, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        imgs.append(img)

    return np.array(imgs)

#Runs inference using the trained SVM model on session character images
def infer_svm(session_path, model_pth=model_path):
    result_data = "svm inference"

    if not os.path.exists(model_pth):
        raise FileNotFoundError(f"SVM model not found: {model_pth}")

    bundle = pickle.load(open(model_pth, "rb"))
    svm = bundle["model"]
    names = bundle["class_names"]

    imgs = _load_chars_for_inference(session_path)
    if imgs.shape[0] == 0:
        return result_data, "", []

    X = flatten_images(imgs)

    probs = svm.predict_proba(X)
    preds = probs.argmax(axis=1)

    confidences = probs.max(axis=1).round(3).tolist()
    plate = "".join(names[i] for i in preds)

    return result_data, plate, confidences

#Manual test run
if __name__ == "__main__":
    train_svm()
