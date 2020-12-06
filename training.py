import os

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
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from helpers import evaluate, get_class_distribution, train

dirname = os.path.dirname(os.path.abspath(__file__))

FILE_DIR = os.path.join(dirname, "dataset/256x256")
CSV_PATH = "dataset/image-labels.csv"
BATCH_SIZE = 10
EPOCHS = 10
MODEL_PATH = "model_state.pth"
CLASSES = ("solo", "pair", "triple", "quad", "penta +")
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Adjust imbalanced dataset classes
df = pd.read_csv("dataset/image-labels.csv", header=None)


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        # self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        # self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.image_arr = np.asarray(images)
        # Second column is the labels
        self.label_arr = np.asarray(labels)
        # Calculate len
        self.data_len = len(self.label_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(FILE_DIR + "/" + single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


# split dataset into train,test,val
# change line 37
#

images = df.iloc[:, 0]
labels = df.iloc[:, 1]

imgFull, imgTest, labelFull, labelTest = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=69
)

imgTrain, imgVal, labelTrain, labelVal = train_test_split(
    images, labels, test_size=0.1, stratify=labels, random_state=42
)


train_dataset = CustomDataset(imgTrain, labelTrain)
test_dataset = CustomDataset(imgTest, labelTest)
val_dataset = CustomDataset(imgVal, labelVal)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

"""
I would use a stratified split using e.g. sklearn.model_selection.train_test_split,
calculate the weights separately for each split, and use a WeightedRandomSampler for
each subset to get balanced batches.
"""
# Create Dataset
# train_dataset = CustomDataset(CSV_PATH)

# class_counts = df.iloc[:, 1].value_counts().sort_index()
# num_samples = sum(class_counts)

# class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
# weights = [class_weights[labelTrain[i]] for i in range(int(num_samples))]
# sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

# # Create sampler for Dataloader
# sampler = WeightedRandomSampler(
#     weights=class_weights, num_samples=len(class_weights), replacement=True
# )

# print(len(train_loader))
# for idx, (img, label) in enumerate(test_loader):
#     print(idx, label)


class Net(nn.Module):
    def __init__(self, train_CNN=True, num_classes=4):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 126, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(468846, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 468846)
        x = self.fc1(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load in model (check to see if it exists)
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

print("eval pre train")
evaluate(model, val_loader, CLASSES)

# Train model
train(model, train_loader, EPOCHS, LEARNING_RATE, MOMENTUM)

print("eval post train")
evaluate(model, val_loader, CLASSES)

print("Saving model state")
torch.save(model.state_dict(), MODEL_PATH)
