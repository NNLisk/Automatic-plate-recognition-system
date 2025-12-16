import os
import sys
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as f
from PIL import Image

#gives project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.training.Random_forest import infer_random_forest
from src.preprocessing.preprocessing import get_cropped_plate, process_cropped, thresholded_2_segmented_letters, segment_and_file_letters
from src.utils.filer import make_new_session
from src.training.customCNN import convolutional_neural_network
from src.training.KNN import inferKNN
from src import config

device = config.device



def processImage(sessionPath, sessionID, model):

    # sessionpath includes the id, id just for reference
    # sessionPath, sessionID = make_new_session()
    # inputFileName = os.path.join("data", "inference", "raw", filename)
    # img = cv2.imread(inputFileName, cv2.IMREAD_COLOR)

    # if img is None:
    #     raise Exception(f"Could not read image: {inputFileName}")
    # # saves the raw image
    # cv2.imwrite(os.path.join(sessionPath, "rawinput.jpg"), img)
    result_data = "Saved to a new session: " + sessionPath

    cropped = get_cropped_plate(os.path.join(sessionPath, "rawinput.jpg"), sessionPath)
    thresheld = process_cropped(os.path.join(sessionPath, "cropped.jpg"), sessionPath)
    
    contours, rd = thresholded_2_segmented_letters(os.path.join(sessionPath, "thresholded.jpg"), sessionPath)
    result_data += rd

    segment_and_file_letters(sessionPath, contours)

    if model == "CNN":
        plate, rd2, confidence_values = inferCharacterCNN(sessionPath)
        
    if model == "KNN":
        plate, rd2, confidence_values = inferCharactersKNN(sessionPath)

    #random forest model switch
    if model == "RFC":
        plate, rd2, confidence_values = inferCharactersRF(sessionPath)
        

    result_data += rd2
    result_data += "\n#########################\n"
    return result_data, plate, confidence_values
    


    

def inferCharacterCNN(sessionPath):
    model = convolutional_neural_network()
    model.load_state_dict(torch.load(os.path.join("models", "CNN", "character_cnn_best.pth"), map_location=torch.device(device)))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((100,75)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = os.path.join(sessionPath, "characters")

    characters = len([name for name in os.listdir(path=path) if os.path.isfile(os.path.join(path, name))])
    result_data = "Characters found: " + str(characters)
    
    plate = ""
    confidences = []
    for i in range(characters):
        img = Image.open(os.path.join(path, f"{i+1}.jpg"))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            ## INFERENCE PART
            output = model(img_tensor)
            ## softmax added to get confidence values
            probabilities = f.softmax(output, dim=1)

            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidences.append(round(confidence.item(), 3))
        class_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

        plate += (class_names[predicted_class])
    return plate, result_data, confidences



def inferCharactersKNN(sessionPath):
    plateindices, confidences = inferKNN(sessionPath)
    class_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

    plate = ""
    
    for i in plateindices:
        plate += class_names[i]

    return plate, "x", confidences

def inferCharactersRF(sessionPath):
    rd, plate, confidences = infer_random_forest(sessionPath)
    return plate, rd, confidences
    

if __name__ == "__main__":
    #pipeline: rawimage filename -> session folder automatically with output in terminal
    spath, sid = make_new_session()
    processImage(spath, sid)
