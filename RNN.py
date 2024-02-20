# Import & Setup

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transform
from torchsummary import summary

# Hyper-Parameters

epochs = 2
batch_size = 64
learning_rate = 0.01
classes = 10
num_inputs = 28
sequence_length = 28
num_layer =2
num_hidden = 256

# Dataset Preparation

train_dataset = torchvision.datasets.MNIST(root="\data", train=True, transform=transform.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="\data", train=False, transform=transform.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Model Building
class RNN(nn.Module):
    def __init__(self,num_inputs,num_hidden,num_layer,classes):
        super(RNN,self).__init__()
        self.num_layer = num_layer
        self.num_hidden = num_hidden
        self.rnn = nn.RNN(num_inputs,num_hidden,num_layer,batch_first=True)
        self.fc = nn.Linear(num_hidden*sequence_length,classes)

    def forward(self,x):
        ho = torch.zeros(self.num_layer,x.size(0),self.num_hidden)
        # forward pass

        out, _ = self.rnn(x,ho)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out


Model = RNN(num_inputs, num_hidden,num_layer, classes)
total_params = sum(p.numel() for p in Model.parameters())
print(f"Total trainable parameters: {total_params}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), learning_rate)

# Training Model

n_step = len(train_loader)

for i in range(epochs):

    for j, (image, label) in enumerate(train_loader):
        image = image.squeeze(1) # Flatting

        # Forward Pass

        output = Model(image)
        loss = criterion(output, label)

        # Backward Propagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j + 1) % 100 == 0:
            print(f"Epochs [{i + 1}/{epochs}], Step [{j + 1}/{n_step}], loss:{loss.item()}")

with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for image, label in test_loader:
        image = image.squeeze(1)
        output = Model(image)

        # Value_Index

        _, prediction = torch.max(output, -1)
        n_sample += label.shape[0]
        n_correct += (prediction == label).sum().item()

accuracy = 100 * n_correct / n_sample

print(f"Accuracy = {accuracy}")
