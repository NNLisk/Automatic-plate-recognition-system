import os
import sys
import cv2
import torch
from torchvision import transforms
from PIL import Image

#gives project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import get_cropped_plate, process_cropped, thresholded_2_segmented_letters, segment_and_file_letters
from src.utils.filer import make_new_session
from src.training.customCNN import convolutional_neural_network




def processImage(filename):

    # sessionpath includes the id, id just for reference
    sessionPath, sessionID = make_new_session()
    inputFileName = f"data/inference/raw/{filename}"
    img = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception(f"Could not read image: {inputFileName}")
    # saves the raw image
    cv2.imwrite(f"{sessionPath}/rawinput.jpg", img)

    cropped = get_cropped_plate(f"{sessionPath}/rawinput.jpg", sessionPath)
    thresheld = process_cropped(f"{sessionPath}/cropped.jpg", sessionPath)
    
    contours = thresholded_2_segmented_letters(f"{sessionPath}/thresholded.jpg", sessionPath)

    segment_and_file_letters(sessionPath, contours)

    

def inferCharacter(sessionPath):
    model = convolutional_neural_network()
    model.load_state_dict(torch.load('models/CNN/v2/characters_cnn_best_pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((100,75)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = f"{sessionPath}/characters/"

    characters = len([name for name in os.listdir(path=path) if os.path.isfile(name)])

    for i in range(characters):
        pass


if __name__ == "__main__":
    
    processImage("test2.jpg")