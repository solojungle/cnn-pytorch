import os
from collections import defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

dirname = os.path.dirname(os.path.abspath(__file__))

# Hyper parameters
FILE_DIR = os.path.join(dirname, "dataset/256x256")
CLASSES = ("1", "2", "3", "4", "5+")
NUM_CLASSES = len(CLASSES)

"""
classes.py

* Had to update multiple files in order to change the architecture of the Net
* Decided to abstract all classes into a single file to make things simpler
"""


class AlbumentationsDataset(Dataset):
    """
    AlbumentationsDataset

    * Uses PIL to open images because had KeyError using other libraries
    * Performs transformation/augmentation on image before returning
    """

    # Constructor
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = np.asarray(file_paths)
        self.labels = np.asarray(labels)
        self.transform = transform

    # Returns length of dataset
    def __len__(self):
        return len(self.file_paths)

    # Returns image and label, performs transformation on image if exists
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
    """
    Net
    * Loosely based off of VGG16
    * Originally planned on training with images of size 256x256 however,
      was too large to fit inside my GPU
    * Dropout is used to try and prevent model from overfitting
    """

    def __init__(self, train_CNN=True, num_classes=NUM_CLASSES):
        super(Net, self).__init__()
        # ((tensorW-kernelWidth+2*Padding)/Stride)+1
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    # Evaluates a batch of images on the network
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = x.view(-1, 4096)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.out(x)

        return x


class MetricMonitor:
    """
    MetricMonitor

    * Provide an aesthetic way of representing training progress
    * Display running accuracy/error on training set
    * Includes a progress bar
    """

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
