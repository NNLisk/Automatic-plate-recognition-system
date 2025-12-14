import os, sys, cv2, joblib
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Resolve project root so imports and relative paths work when run directly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

# Image and label settings shared with other OCR models
dimension = (100, 75)  # resize target (w, h)
class_names = ['0','1','2','3','4','5','6','7','8','9',
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
               'P','Q','R','S','T','U','V','W','X','Y','Z']
model_path = os.path.join("models", "RFC", "rf_ocr.joblib")

def _load_images(root):
    # Load all class folders under root into arrays; labels are folder indices.
    folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    imgs, labels = [], []
    for idx, d in enumerate(folders):
        print("LOADING IMAGES")
        for fn in os.listdir(os.path.join(root, d)):
            fp = os.path.join(root, d, fn)
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            labels.append(idx)
    return np.array(imgs), np.array(labels)

def _extract_features(imgs):
    # Compute HOG features for each image; produces 1D feature vectors.
    print("HOG")
    feats = []
    for img in imgs:
        feats.append(hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True))
    return np.array(feats, dtype=np.float32)

def train_rf(data_root=os.path.join("data", "OCR_training_data", "data", "train")):
    # Train RandomForest on the OCR dataset, report metrics, and save model+scaler.
    imgs, labels = _load_images(data_root)
    X = _extract_features(imgs)
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.1, stratify=labels, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    
    rf.fit(Xtr, ytr)
    preds = rf.predict(Xte)
    print(f"Accuracy: {accuracy_score(yte, preds):.3f}")
    print(classification_report(yte, preds, digits=3))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": rf, "scaler": scaler, "class_names": class_names}, model_path)
    return rf

def _load_chars_for_inference(session_path):
    # Load segmented character images from a session folder.
    chpath = os.path.join(session_path, "characters")
    files = sorted(f for f in os.listdir(chpath) if os.path.isfile(os.path.join(chpath, f)))
    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(chpath, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        imgs.append(img)
    return np.array(imgs)

def infer_random_forest(session_path, model_pth=model_path):

    result_data = "kakka"
    # Load saved RF model, run inference on session characters, return plate and confidences.
    bundle = joblib.load(model_pth)
    rf, scaler, names = bundle["model"], bundle["scaler"], bundle["class_names"]
    imgs = _load_chars_for_inference(session_path)
    feats = _extract_features(imgs)
    probs = rf.predict_proba(scaler.transform(feats))
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1).round(3).tolist()
    plate = "".join(names[i] for i in preds)
    return rd, plate, confidences

if __name__ == "__main__":
    # Run training when executed directly
    train_rf()
