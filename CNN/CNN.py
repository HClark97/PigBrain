# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:54:19 2022

@author: HClark
"""

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import h5py
import matplotlib as plt
import mpu
import plyer as pl
import torchvision

'''### Device configuration ###'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''### Hyperparameters ###'''
batch_size = 32
epochs = 5

'''### Data ###'''
path = pl.filechooser.choose_dir()
#nox_dataset = torch.load('nox.pt')

train_data = torchvision.datasets.DatasetFolder(root=path[0],loader='torch.load')

# path = pl.filechooser.choose_dir()
# nonnox_dataset = torch.load('nonnox.pt')

# path = pl.filechooser.open_file()
# nox_dataset = mpu.io.read('nox.pickle')
# path = pl.filechooser.open_file()
# nonnox_dataset = mpu.io.read('nonnox.pickle')
# train_dataset = [nox_dataset, nonnox_dataset]
#KongeKat = ConcatDataset(train_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# path = pl.filechooser.open_file()
# nox_dataset = mpu.io.read('nox.pickle')
# path = pl.filechooser.open_file()
# nonnox_dataset = mpu.io.read('nonnox.pickle')
# val_dataset = [nox_dataset, nonnox_dataset]
# KongeHund = ConcatDataset(val_dataset)
# val_loader = DataLoader(KongeHund, batch_size=batch_size, shuffle=True)

# test_dataset = MNIST('/files/', train=False, download=True, transform=ToTensor())
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


'''### Model definition ###'''
### Define architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)                       # -> n, 10
        
        return x
    
### Instantiate the network
model = ConvNet().to(device)
### Define the optimizer
optimizer = optim.Adam(model.parameters())
### Define the loss function
criterion = nn.BCELoss()

'''### Training ###'''
train_loss_history = list()
n_correct = 0
n_samples = 0
### Run through epochs
for epoch in range(epochs):
    ### Run through batches
    train_loss = 0.0
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

        ### Save losses
        train_loss += loss.item()
        
        ### Print Epoch, batch and loss
        if i % 40 == 39:  # print every 2000 mini-batches
            print('[Epoch: {} Batch: {}/{}] loss: {}'.format(
                  epoch + 1, i + 1, len(train_loader), train_loss / 2000))
    ### Save loss in history        
    train_loss = train_loss/len(train_loader)
    train_loss_history.append(train_loss)


### Save model
# torch.save(model.state_dict(), FILEPATH)



'''### Validation model ###'''
model = ConvNet().to(device)
# model.load_state_dict(torch.load("Filename"))
model.eval() 
val_loss_history = list()
for epoch in range(epochs):
    val_loss = 0.0
    for i, (imgs, labels) in enumerate(val_loader):
        prediction = model(imgs)
        val_loss += criterion(prediction, labels).item()
        
    val_loss = val_loss/len(val_loader)
    val_loss_history.append(val_loss)
        

'''### Plot test and validation model ###'''
plt.plot(train_loss_history, label='train')        
plt.plot(val_loss_history, label='validation')
plt.title('Loss model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()



# '''### Testing model ###'''
# model = ConvNet().to(device)
# model.load_state_dict(torch.load("Filename"))
# n_correct = 0
# n_samples = 0
# with torch.no_grad():
#     for images, labels in  enumerate(test_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#     accuracy = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network: {accuracy} %')