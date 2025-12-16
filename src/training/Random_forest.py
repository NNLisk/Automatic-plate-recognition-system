import os, sys, cv2, joblib
import numpy as np
import random
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
model_path = os.path.join("models", "RFC", "v4", "rf_ocr.joblib")

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

def augment_char(img):
    # Character augmentations to improve generalization.
    h, w = img.shape[:2]

    # small rotations
    angle = random.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # slight translation
    tx, ty = random.randint(-2, 2), random.randint(-2, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # brightness/contrast jitter (keep in 0–255)
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.randint(-20, 20)    # brightness
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    # slight Gaussian blur or noise (randomly chosen)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.4:
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

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

    # Split before augmentation so only the training fold is augmented.
    im_tr, im_te, ytr, yte = train_test_split(
        imgs, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # In-memory augmentation 
    aug_imgs, aug_labels = [], []
    for img, lbl in zip(im_tr, ytr):
        aug = augment_char((img * 255).astype(np.uint8))
        aug_imgs.append(aug.astype(np.float32) / 255.0)
        aug_labels.append(lbl)
    im_tr = np.concatenate([im_tr, np.array(aug_imgs, dtype=np.float32)])
    ytr = np.concatenate([ytr, np.array(aug_labels)])

    Xtr = _extract_features(im_tr)
    Xte = _extract_features(im_te)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # v3
    #rf = RandomForestClassifier(
    #    n_estimators=500,
    #    max_depth=None,
    #    n_jobs=-1,
    #    class_weight="balanced_subsample",
    #    random_state=42,
    #)
    
    # v4+
    rf = RandomForestClassifier(
    n_estimators=200,          # start 100–300
    max_depth=20,              # start 12–25
    min_samples_leaf=3,        # start 2–10
    min_samples_split=6,       # start 4–20
    max_leaf_nodes=5000,
    max_features="sqrt",
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
    return result_data, plate, confidences

if __name__ == "__main__":
    # Run training when executed directly
    train_rf()
