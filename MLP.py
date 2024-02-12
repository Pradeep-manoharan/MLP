# Import & Setup

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform

# Hyper-Parameters

epochs = 2
batch_size = 100
learning_rate = 0.01
classes = 10
num_inputs = 28 * 28
num_hidden = 500

# Dataset Preparation

train_dataset = torchvision.datasets.MNIST(root="\data", train=True, transform=transform.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="\data", train=False, transform=transform.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Model Building

class mlp(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_classes):
        super(mlp,self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.L1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        out = self.L1(x)
        out = self.relu(out)
        out = self.L2(out)

        return out


Model = mlp(num_inputs, num_hidden, classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(),learning_rate)

# Training Model

n_step = len(train_loader)

for i in range(epochs):

    for j, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, 28 * 28)  # Flatting

        # Forward Pass

        output = Model(image)
        loss = criterion(output, label)

        # Backward Propagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(j+1) % 100 == 0:
            print(f"Epochs [{i+1}/{epochs}], Step [{j+1}/{n_step}], loss:{loss.item()}")

with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for image,label in test_loader:

        image = image.reshape(-1, 28*28)
        output = Model(image)

        # Value_Index

        _, prediction = torch.max(output,-1)
        n_sample += label.shape[0]
        n_correct += (prediction == label).sum().item()

accuracy = 100 * n_correct / n_sample

print(f"Accuracy = {accuracy}")