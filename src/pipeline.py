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
from src import config

device = config.device



def processImage(filename):

    # sessionpath includes the id, id just for reference
    sessionPath, sessionID = make_new_session()
    inputFileName = os.path.join("data", "inference", "raw", filename)
    img = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception(f"Could not read image: {inputFileName}")
    # saves the raw image
    cv2.imwrite(os.path.join(sessionPath, "rawinput.jpg"), img)

    cropped = get_cropped_plate(os.path.join(sessionPath, "rawinput.jpg"), sessionPath)
    thresheld = process_cropped(os.path.join(sessionPath, "cropped.jpg"), sessionPath)
    
    contours = thresholded_2_segmented_letters(os.path.join(sessionPath, "thresholded.jpg"), sessionPath)

    segment_and_file_letters(sessionPath, contours)
    print("Saved to a new session: " + sessionPath)

    plate = inferCharacter(sessionPath)

    print(f"Recognized plate: {plate}")

    print("\n#########################\n")


    

def inferCharacter(sessionPath):
    model = convolutional_neural_network()
    model.load_state_dict(torch.load(os.path.join("models", "CNN", "v4", "character_cnn_best.pth"), map_location=torch.device(device)))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((100,75)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = os.path.join(sessionPath, "characters")

    characters = len([name for name in os.listdir(path=path) if os.path.isfile(os.path.join(path, name))])
    print("Characters found: " + str(characters))
    
    plate = ""

    for i in range(characters):
        img = Image.open(os.path.join(path, f"{i+1}.jpg"))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
        class_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

        plate += (class_names[predicted_class])
    return plate

if __name__ == "__main__":
    #pipeline: rawimage filename -> session folder automatically with output in terminal
    processImage("test_california.jpg")
