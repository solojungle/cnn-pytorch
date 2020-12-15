import os
import time

import albumentations as A
import cv2
import matplotlib.patches as patches
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
from matplotlib import pyplot as plt
from PIL import Image
from skimage import color, data
from skimage.transform import downscale_local_mean, rescale, resize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from classes import AlbumentationsDataset, Net
from helpers import evaluate, get_class_info, get_sampler_weights, load_model, train

"""
cleandata.py

* Display the first 64 images that were predicted wrong in the dataset
* Was used to correct image labels
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/1607868216-model.pth"

NUM_OF_IMAGES = 5063

dirname = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(dirname, "dataset/256x256")

df = pd.read_csv("dataset/image-labels.csv", header=None)
imagesDF = df.iloc[:, 0]
labelsDF = df.iloc[:, 1]

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)


name_of_class = ["1", "2", "3", "4", "5+"]


val_transform = A.Compose(
    [
        A.Resize(64, 64),
        A.Normalize(),
        ToTensorV2(),
    ]
)


class CleanDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = np.asarray(file_paths)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def get_file_path(self, idx):
        return self.file_paths[idx]

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        pillow_image = Image.open(FILE_DIR + "/" + file_path)
        image = np.array(pillow_image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


train_dataset = CleanDataset(imagesDF, labelsDF, transform=val_transform)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=10,
)

torch.cuda.empty_cache()

model = load_model(MODEL_PATH)


all_predictions = []
all_labels = []

with torch.no_grad():
    for imgs, labels in train_loader:

        # data to cuda
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)

        # Append predictions for confusion matrix
        all_predictions.append(predicted.cpu().numpy())
        all_labels.append(labels.cpu().numpy())


all_predictions = [a.squeeze().tolist() for a in all_predictions]
all_predictions = [val for sublist in all_predictions for val in sublist]

all_labels = [a.squeeze().tolist() for a in all_labels]
all_labels = [val for sublist in all_labels for val in sublist]


"""
Gets the first 64 images that were incorrect, shows the label, filename and prediction
then plots them all with matlab
"""
i = 0
j = 0
while j < 63:
    if i > NUM_OF_IMAGES:
        break
    if all_predictions[i] != all_labels[i]:
        j += 1
        ax = fig.add_subplot(8, 8, j + 1, xticks=[], yticks=[], facecolor="black")

        file_path = train_dataset.get_file_path(i)

        pillow_image = Image.open(FILE_DIR + "/" + file_path)
        image = np.array(pillow_image)

        image = resize(image, (64, 64))

        ax.imshow(image, cmap=plt.cm.binary, interpolation="nearest")
        # label the image with the target value
        ax.text(
            0, 53, "Guess: " + str(name_of_class[all_predictions[i]]), color="white"
        )
        ax.text(0, 47, "Label: " + str(name_of_class[all_labels[i]]), color="white")
        ax.text(0, 60, file_path, color="white")

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (0, 40), 64, 64, linewidth=1, facecolor="black", alpha=0.2
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    i += 1


plt.show()
