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

from classes import AlbumentationsDataset, Net
from helpers import evaluate, get_class_info, get_sampler_weights, load_model, train

dirname = os.path.dirname(os.path.abspath(__file__))

# Hyper parameters
FILE_DIR = os.path.join(dirname, "dataset/256x256")
CSV_PATH = "dataset/image-labels.csv"
BATCH_SIZE = 15
EPOCHS = 375
CLASSES = ("1", "2", "3", "4", "5+")
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Generate a new model file
# now = int(time.time())
# MODEL_PATH = "models/" + str(now) + "-" + "model.pth"

MODEL_PATH = "models/1607949386-model.pth"

# Declare an augmentation pipeline
transform = A.Compose(
    [
        A.Resize(64, 64),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToGray(p=0.2),
        # https://discuss.pytorch.org/t/expected-object-of-scalar-type-byte-but-got-scalar-type-float/66462
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(64, 64),
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


# Create sampler
train_weights = get_sampler_weights(train_dataset)
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

# Create sampler
val_weights = get_sampler_weights(val_dataset)
val_sampler = WeightedRandomSampler(val_weights, len(val_weights))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
)

# Clear cache
torch.cuda.empty_cache()

model = load_model(MODEL_PATH)

train(model, train_loader, EPOCHS, LEARNING_RATE, MOMENTUM)

print("Saving model state as: ", MODEL_PATH, "\n")
torch.save(model.state_dict(), MODEL_PATH)

print("Number of classes in training dataset")
num_of_classes, _ = get_class_info(train_dataset)
print(num_of_classes)

print("Evaluating model...")
evaluate(model, val_loader, CLASSES)
