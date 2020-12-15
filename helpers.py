import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from classes import MetricMonitor, Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(output, target):
    """
    calculate_accuracy
    * Helper function that returns the accuracy given a prediction and label
    """
    _, predicted = torch.max(output.data, 1)
    return 100 * (predicted == target).sum() / target.size(0)


def train(model, dataloader, epochs, learningRate=0.001, momentumRate=0.9):
    """
    train
    * Trains the net, sends progress to a MetricMonitor instance
    * Could optimize train by accepting an object of params instead of taking them individually
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentumRate)

    for epoch in range(1, epochs + 1):
        metric_monitor = MetricMonitor()
        model.train()
        stream = tqdm(dataloader)
        for i, (images, labels) in enumerate(stream, start=1):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            accuracy = calculate_accuracy(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor
                )
            )
    print()  # Add newline


def get_confusion_matrix(target, output):
    """
    get_confusion_matrix
    * Creates and displays a confusion matrix
    """
    confusion_matrix_df = pd.DataFrame(confusion_matrix(target, output))
    sns.heatmap(confusion_matrix_df, annot=True)
    plt.show()
    return


def evaluate(model, dataloader, classes):
    """
    evaluate
    * Runs a DataLoader through a Net, and displays performance information
    """
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:

            # data to cuda
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            # Append predictions for confusion matrix
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n####################\n")

    for i in range(len(classes)):
        if class_total[i] == 0:
            print("No classes for ", classes[i])
        else:
            print(
                "Accuracy of %5s : %2d%%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )

    # Flatten the arrays into their respective lists before feeding to functions
    all_predictions = [a.squeeze().tolist() for a in all_predictions]
    all_predictions = [val for sublist in all_predictions for val in sublist]
    all_labels = [a.squeeze().tolist() for a in all_labels]
    all_labels = [val for sublist in all_labels for val in sublist]

    print("\n####################\n")

    print(classification_report(all_labels, all_predictions))

    get_confusion_matrix(all_labels, all_predictions)


def load_model(model_path):
    """
    load_model
    * Checks to see if model.pth exists, if not create a new one with the given name
    """

    model = None
    if os.path.isfile(model_path):
        print("Loading model state...")
        model = Net().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("Creating new model...")
        model = Net().to(device)

    return model


def get_sampler_weights(dataset):
    """
    get_sampler_weights
    * Given a dataset, calculates the class distribution, and creates a weight for each image
    """

    # Count the number of classes
    class_sample_count, targets = get_class_info(dataset)
    # Create a weight distribution
    weight = 1.0 / class_sample_count.float()
    # Give a weight to every image in the dataset
    samples_weight = torch.tensor([weight[t] for t in targets])

    return samples_weight


def get_class_info(dataset):
    """
    get_class_info
    * Returns the number of classes, and a list of its labels
    """
    targets = []
    for _, target in dataset:
        targets.append(target)

    targets = torch.tensor(targets)
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )

    return class_sample_count, targets
