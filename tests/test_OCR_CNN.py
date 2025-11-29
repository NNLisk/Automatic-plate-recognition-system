import torch
import cv2
from torchvision import transforms
from PIL import Image
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.training.customCNN import convolutional_neural_network

model = convolutional_neural_network()
model.load_state_dict(torch.load('models/CNN/v2/character_cnn_best.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((100,75)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open('data/inference/sessions/2/characters/7.jpg')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

print(f"Predicted class: {predicted_class}")

class_names = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
print(f"predicted character: {class_names[predicted_class]}")