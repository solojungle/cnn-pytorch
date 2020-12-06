import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, epochs, learningRate=0.001, momentumRate=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentumRate)

    print("Beginning training...")
    for e in range(epochs):

        losses = []
        accuracies = []

        model.train()
        for imgs, labels in dataloader:
            # data to cuda
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc running training accuracy
            _, predictions = outputs.max(1)
            num_correct = (predictions == labels).sum()

        print(f"Mean Loss this epoch was {sum(losses)/len(losses)}")


def get_class_distribution(obj):
    count_dict = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }

    for i in obj:
        if i == 0:
            count_dict["0"] += 1
        elif i == 1:
            count_dict["1"] += 1
        elif i == 2:
            count_dict["2"] += 1
        elif i == 3:
            count_dict["3"] += 1
        elif i == 4:
            count_dict["4"] += 1

    return count_dict


def evaluate(model, dataloader, classes):
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for imgs, labels in dataloader:

            # data to cuda
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] == 0:
            print("No classes for ", classes[i])
        else:
            print(
                "Accuracy of %5s : %2d%%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )


def load_model(model_path):
    # Load in model (check to see if it exists)
    model = None
    if os.path.isfile(model_path):
        print("Loading model state...")
        model = Net().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        # Create new model
        print("Creating new model...")
        model = Net().to(device)

    return model
