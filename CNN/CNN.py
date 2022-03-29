from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import h5py

'''### Hyperparameters ###'''
batch_size = 32
epochs = 5

'''### Data ###'''
train_dataset = MNIST('/files/', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNIST('/files/', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

'''### Model definition ###'''
### Define architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.linear1 = nn.Linear(in_features=64*28//4*28//4, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        return x
### Instantiate the network
model = Net()
### Define the optimizer
optimizer = optim.Adam(model.parameters())
### Define the loss function
criterion = nn.CrossEntropyLoss()

'''### Training ###'''
### Run through epochs
for epoch in range(epochs):
    ### Run through batches
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        ### Zero the gradients of the network
        optimizer.zero_grad()
        ### Run the batch through the model to get the predictions
        prediction = model(imgs)
        ### Calculate the loss between labels and prediction
        loss = criterion(prediction, labels)
        ### Do backpropagation to calculate gradients
        loss.backward()
        ### Make a step with the optimizer
        optimizer.step()

        ### Print progress
        running_loss += loss.item()
        if i % 40 == 39:  # print every 2000 mini-batches
            print('[Epoch: {} Batch: {}/{}] loss: {}'.format(
                  epoch + 1, i + 1, len(train_loader), running_loss / 2000))
            running_loss = 0.0


correct = 0
for i, (imgs, labels) in enumerate(test_loader):
    outputs = model(imgs)
    prediction = torch.argmax(outputs.data, 1)
    correct += (prediction == labels).float().sum()
accuracy = correct / len(test_dataset)
print("Accuracy = {}".format(accuracy))