import sys, os
import cv2
import numpy as np
from math import floor


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

dimension = (100, 75)

def load_images(containerPath):
    image_directory = containerPath
    
    folders = [d for d in os.listdir(image_directory) if os.path.isdir(os.path.join(image_directory, d))]
    
    folders.sort()
    

    images = []
    labels = []

    description = "An OCR dataset for k-NN algorithm"

    # preprocessing
    for i, d in enumerate(folders):
        fpath = os.path.join(image_directory, d)
        
        for file in os.listdir(fpath):
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


def process(x):
    nsamples = x.shape[0]
    imgSize = x.shape[1] * x.shape[2]
    

    flattened = np.zeros((nsamples, imgSize), dtype=np.float32)
    for i, img in enumerate(x):
        flattened[i] = img.flatten()
    
    return flattened

def split(x, y):

    s = x.shape[0]
    eighty = floor(s * 0.80)
    # print(eighty)
    # print(s - eighty)
    # print(s)
    # print(x.size)

    xtrain, xtest, ytrain, ytest = x[:eighty], x[eighty:], y[:eighty], y[eighty:]
    return xtrain, xtest, ytrain, ytest



## the functions take the training images, loads and prepares them, flattens all images and splits into training and testing
## for this i thought just the training set is enough. it has more than 28000 images
x, y = load_images(os.path.join("data", "OCR_training_data", "data", "train"))
xflat = process(x)

xtrain, xtest, ytrain, ytest = split(xflat,y)

# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)

def load_infer_chars(sessionPath):
    chpath = os.path.join(sessionPath, "characters")
    folder = [im for im in os.listdir(chpath) if os.path.isfile(os.path.join(chpath, im))]
    folder.sort()
    print(folder)
    
    imgs = []

    for i, image in enumerate(folder):
        file = os.path.join(chpath, image)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if img is None:
                continue

        img = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) /255.0

        imgs.append(img)

    imagearray = np.array(imgs)

    iarr = process(imagearray)
    return iarr


def inferKNN(sessionPath):
    global xtrain, ytrain
    inferring = load_infer_chars(sessionPath)
    k = 100

    plateindices = []
    confidences = [] 

    for i, image in enumerate(inferring):
        difference = image - xtrain
        distances = np.sqrt(np.sum(difference**2, axis=1))

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = ytrain[nearest_indices]

        classes, amounts = np.unique(nearest_labels, return_counts=True)
        print(classes)
        print(amounts)
        print("#############")
        confidences.append(round(float(amounts[np.argmax(amounts)])/k, 2))
        predicted = classes[np.argmax(amounts)]


        plateindices.append(int(predicted))

    return plateindices, confidences

def euclidean_distance(x, y):
    distance = np.sqrt(np.sum(x-y)**2)
    return distance

if __name__ == "__main__":
    plt, confidences = inferKNN(os.path.join("data", "inference", "sessions", "1"))
    print(plt)
    print(confidences)
