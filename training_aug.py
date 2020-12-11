import os
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from helpers import evaluate, get_class_distribution, train

dirname = os.path.dirname(os.path.abspath(__file__))

# Hyper parameters
FILE_DIR = os.path.join(dirname, "dataset/256x256")
CSV_PATH = "dataset/image-labels.csv"
BATCH_SIZE = 15
EPOCHS = 50
CLASSES = ("solo", "pair", "triple", "quad", "penta")
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# now = int(time.time())
# MODEL_PATH = "models/" + str(now) + "-" + "model.pth"

MODEL_PATH = "models/183853model.pth"


class AlbumentationsDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = np.asarray(file_paths)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        pillow_image = Image.open(FILE_DIR + "/" + file_path)
        image = np.array(pillow_image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class Net(nn.Module):
    def __init__(self, train_CNN=True, num_classes=4):
        super(Net, self).__init__()
        # ((tensorW-kernelWidth+2*Padding)/Stride)+1
        self.conv = nn.Conv2d(3, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(100352, 12544)
        self.fc2 = nn.Linear(12544, 1568)
        self.out = nn.Linear(1568, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 100352)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


# Declare an augmentation pipeline
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # https://discuss.pytorch.org/t/expected-object-of-scalar-type-byte-but-got-scalar-type-float/66462
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2(),
    ]
)


df = pd.read_csv("dataset/image-labels.csv", header=None)
images = df.iloc[:, 0]
labels = df.iloc[:, 1]

imgFull, imgTest, labelFull, labelTest = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=69
)

imgTrain, imgVal, labelTrain, labelVal = train_test_split(
    images, labels, test_size=0.1, stratify=labels, random_state=42
)

train_dataset = AlbumentationsDataset(imgTrain, labelTrain, transform=transform)
val_dataset = AlbumentationsDataset(imgVal, labelVal, transform=val_transform)

# Create sampler (Overfits the model currently)
# class_counts = df[1].value_counts().sort_index()
# class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# sampler = WeightedRandomSampler(class_weights, sum(class_counts))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = None
if os.path.isfile(MODEL_PATH):
    print("Loading model state...")
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
else:
    # Create new model
    print("Creating new model...")
    model = Net().to(device)

# train(model, train_loader, EPOCHS, LEARNING_RATE, MOMENTUM)

# print("Saving model state as: ", MODEL_PATH)
# torch.save(model.state_dict(), MODEL_PATH)

print("Evaluating model...")
evaluate(model, val_loader, CLASSES)
