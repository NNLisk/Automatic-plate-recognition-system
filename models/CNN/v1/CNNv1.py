import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as f
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchmetrics import F1Score, Accuracy, Precision, Recall

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

batch = 64
num_classes = 36
learning_rate = 0.001
num_epochs = 20
channels = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((100,75)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(
    root = 'data/OCR_training_data/data/train/',
    transform = transform
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

test_dataset = datasets.ImageFolder(
    root = 'data/OCR_training_data/data/test/',
    transform = transform
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

val_dataset = datasets.ImageFolder(
    root = 'data/OCR_training_data/data/val/',
    transform = transform
)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)



class convolutional_neural_network(nn.Module):
    def __init__(self):
        super(convolutional_neural_network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*25*18, num_classes)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape) confirms the output has right amount of channels: 7200
        x = self.fc1(x)

        return x


def trainOCR():

    model = convolutional_neural_network().to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        epoch_loss = 0

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
       
            
    torch.save(model.state_dict(), 'models/CNN/character_cnn.pth')
    

if __name__ == "__main__":
    trainOCR()

#== APPENDIX =================================

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()

# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# labels
# imshow(torchvision.utils.make_grid(images))