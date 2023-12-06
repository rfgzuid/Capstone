'''MODEL
Create and train a NNDM
The architecture and other hyperparameters have not been tuned yet

Note: better generalized performance was achieved by making the NNDM
predict the difference between previous and next state Δs instead of next state s[t+1]'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dynamics_data = np.load(r'Cartpole data.npy')
batch_size = 64


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx, :5], self.data[idx, 5:]
        return sample


dataset = CustomDataset(dynamics_data)
train_set, val_set, test_set = random_split(dataset, [0.75, 0.1, 0.15])  # create train/val/test set

train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


class NNDM(nn.Module):
    def __init__(self):
        super(NNDM, self).__init__()

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 4)

        self.activation = nn.functional.tanh

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        h = self.activation(self.fc4(h))
        h = self.activation(self.fc5(h))
        return self.fc6(h) + x[:,:4]  # no output activation function (output is not restricted to a range)


def train(loader, net, optimizer, criterion):
    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss/len(train_loader)

def evaluate(loader, net, criterion):
    avg_loss = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for data in loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += loss

    return avg_loss/len(test_loader)

# Set the number of epochs to for training
N = 10
lr = 5e-3

epochs = list(range(N))
train_losses = []
val_losses = []
test_losses = []

# Create instance of Network
net = NNDM()

# Create loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

for epoch in tqdm(range(N)):  # loop over the dataset multiple times
    # Train on data
    train_loss = train(train_loader, net, optimizer, criterion)
    train_losses.append(train_loss.item())

    val_loss = evaluate(val_loader, net, criterion)
    val_losses.append(val_loss)

    test_loss = evaluate(test_loader, net, criterion)
    test_losses.append(test_loss)

plt.plot(epochs, train_losses, c='r', label='train')
plt.plot(epochs, val_losses, c='y', label='val')
plt.plot(epochs, test_losses, c='b', label='test')
plt.legend()
plt.show()

# save the NNDM to be used in 'NNDM visualizer.py'
torch.save(net.state_dict(), 'NNDM.pt')
