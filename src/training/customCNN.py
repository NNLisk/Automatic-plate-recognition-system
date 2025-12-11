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

from src import config

device = config.device

batch = 64
num_classes = 36
learning_rate = 0.001
num_epochs = 100
channels = 1

transform = None
train_dataset, train_loader = None, None
test_dataset, test_loader = None, None
val_dataset, val_loader = None, None

train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []


def prepare_data():

    transform = transforms.Compose([
        transforms.Resize((100,75)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.3, 0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    global train_dataset, train_loader, test_dataset, test_loader, val_dataset,val_loader

    train_dataset = datasets.ImageFolder(
        root = os.path.join("data", "OCR_training_data", "data", "train"),
        transform = transform
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)

    test_dataset = datasets.ImageFolder(
        root = os.path.join("data", "OCR_training_data", "data", "test"),
        transform = transform
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

    val_dataset = datasets.ImageFolder(
        root = os.path.join("data", "OCR_training_data", "data", "val"),
        transform = transform
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=False)


class convolutional_neural_network(nn.Module):
    def __init__(self):
        super(convolutional_neural_network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        ## with two pooling times, images are halved twice, 100x75 => 25x18
        self.fc1 = nn.Linear(32*25*18, num_classes)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        #x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape) confirms the output has right amount of channels: 7200
        x = self.fc1(x)

        return x

## SEPARATE VALIDATION FUNCTION
def validate(model, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    global val_accuracies, val_losses

    with torch.no_grad():
        for character_data, target_values in val_loader:
            character_data = character_data.to(device)
            target_values = target_values.to(device)
            
            outputs = model(character_data)
            loss = criterion(outputs, target_values)
            val_loss += loss.item()

            # for validation i just went with leaner metrics
            # no F1 scores or anything
            _, predicted = torch.max(outputs, 1)
            total += target_values.size(0)
            correct += (predicted == target_values).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = correct / total

        val_accuracies.append(accuracy)
        val_losses.append(avg_loss)
        return avg_loss, accuracy



def trainOCR():

    model = convolutional_neural_network().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    global train_accuracies, train_losses, val_accuracies, val_losses
    total = 0
    correct = 0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        epoch_loss = 0
        train_accuracy = 0
        

        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            # print("scores: ", scores)
            loss = criterion(scores, targets)
            # print("loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print("epoch loss: ", epoch_loss)

            _, predicted = torch.max(scores, 1)
            # print("predicted: ", predicted)
            total += targets.size(0)
            # print("total: ", total)
            correct += (predicted == targets).sum().item()
            # print("correct: ", correct)
        
        avg_loss = epoch_loss / len(train_loader)

        train_accuracy = correct/total
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        
        # ADDED VALIDATION AFTER EACH EPOCH
        val_loss, val_acc = validate(model, criterion)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "CNN", "character_cnn_best.pth"))

    torch.save(model.state_dict(), os.path.join("models", "CNN", "character_cnn_last.pth"))     
    
    
    # print(train_losses)
    # print(train_accuracies)
    # print(val_losses)
    # print(val_accuracies)
    showmetrics(train_losses, train_accuracies, val_loss, val_accuracies)



def showmetrics(train_losses, train_accuracies, val_loss, val_accuracies):

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.title("CNN training metrics by epoch")

    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')

    axes[0, 1].plot(train_accuracies)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')

    axes[1, 0].plot(val_losses)
    axes[1, 0].set_title('Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')

    axes[1, 1].plot(val_accuracies)
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def testOCR(model):

    model.eval()

    f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    accuracy = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
    recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            f1.update(predicted, targets)
            accuracy.update(predicted, targets)
            precision.update(predicted, targets)
            recall.update(predicted, targets)
    
    print(f"F1 score: {f1.compute():.4f}")
    print(f"accuracy: {accuracy.compute():.4f}")
    print(f"precision: {precision.compute():.4f}")
    print(f"recall: {recall.compute():.4f}")

    return accuracy.compute(), f1.compute(), precision.compute(), recall.compute()
    

if __name__ == "__main__":
    prepare_data()
    trainOCR()

    model = convolutional_neural_network().to(device)

    # model.load_state_dict(torch.load(os.path.join("models", "CNN", "v3", "character_cnn_best.pth")))
    # testOCR(model)

#== APPENDIX =================================

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()

# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# labels
# imshow(torchvision.utils.make_grid(images))