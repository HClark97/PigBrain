# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import h5py
import matplotlib.pyplot as plt
#import mpu
import plyer as pl
import torchvision
import numpy as np

'''### Device configuration ###'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
'''### Hyperparameters ###'''
batch_size = 50
minibatch = 100
epochs = 500
learning_rate = 0.0001

'''### Data ###'''
def torch_loader(path):
    sample = torch.load(path)
    return sample

path=list(['/Users/amaliekoch/Desktop/STFT/Train'])
train_data = torchvision.datasets.DatasetFolder(root=path[0],
                                                loader=torch_loader,
                                                extensions=['.pt'],
                                                )

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

path=list(['/Users/amaliekoch/Desktop/STFT/Val'])
val_data = torchvision.datasets.DatasetFolder(root=path[0],
                                                loader=torch_loader,
                                                extensions=['.pt']
                                                )

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

filepathModel = r'/Users/amaliekoch/Desktop/STFT/models/model.pth'

'''### Model definition ###'''
### Define architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride = 1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0)
        #self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=2, padding=1)
        self.conv1_drop = nn.Dropout2d(0.5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) 
        self.fc1 = nn.Linear(in_features=120, out_features=30)
        self.fc2 = nn.Linear(in_features=30, out_features=2)
        #self.fc3 = nn.Linear(in_features=10, out_features=2)
        self.activation = torch.nn.Softmax(dim=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        #x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv1_drop(self.conv1(x)))) 
        x = self.pool2(F.relu(self.conv2_drop(self.conv2(x))))
        #x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(-1, 120)  
        #x= F.dropout(x, p=0.3, training=self.training)        
        x = self.sigmoid1(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)               
        x = self.sigmoid1(self.fc2(x))
        x = F.dropout(x, p=0.7, training=self.training)               
        #x = self.sigmoid1(self.fc3(x))               
        x = self.activation(x)
        return x
    
    
### Instantiate the network
model = ConvNet().to(device)
### Define the optimizer
optimizer = optim.NAdam(model.parameters(),lr=learning_rate)
### Define the loss function
criterion = nn.CELoss()

'''### Training ###'''
train_loss_history = list()
val_loss_history = list()
n_correct = 0
n_samples = 0
val_best = 100
patience = 0
### Run through epochs
for epoch in range(epochs):
    ### Run through batches
    train_loss = 0.0
    
    for i, (imgs, labels) in enumerate(train_loader):
        labels = torch.tensor(np.eye(2)[np.asarray(labels)],dtype = torch.float32) #one hot encoding, so we got a 32,2 matrix (Alex said this is how it is done)
        imgs, labels = imgs.to(device), labels.to(device)
        ### Zero the gradients of the network, reset gradient numbers
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
        ### Save accuracy
        
        
                ### Print Epoch, batch and loss
        if i % minibatch == minibatch-1:  # print every 40 batches
            print('[Epoch: {} Batch: {}/{}] loss: {}'.format(
                epoch + 1, 
                i + 1, 
                len(train_loader), 
                train_loss / (i*batch_size))) #fix denne algoritme, den virker ikke korrekt
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for i, (imgs, labels) in enumerate(val_loader):
            labels = torch.tensor(np.eye(2)[np.asarray(labels)],dtype = torch.float32) #one hot encoding
            imgs, labels = imgs.to(device), labels.to(device)
            prediction = model(imgs)
            lossVal = criterion(prediction, labels)
            val_loss += lossVal.item()
            ### accuracy of the validation
            n_samples += labels.size(0) #how many samples has it gone through
            n_correct += (torch.round(prediction[:,0]) == labels[:,0]).sum().item() #how many are correct
                        ### Print Epoch, batch and loss
            if i % minibatch == minibatch-1:  # print every 40 batches
                print('[Epoch: {} Batch: {}/{}] loss: {}'.format(
                    epoch + 1, 
                    i + 1, 
                    len(val_loader), 
                    val_loss / (i*batch_size))) #fix denne algoritme, den virker ikke korrekt
    ### Save loss in history        
    train_loss = train_loss/len(train_loader)
    train_loss_history.append(train_loss)
    val_loss = val_loss/len(val_loader)
    val_loss_history.append(val_loss)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
          
    ## Model chechpoint
    if val_loss < val_best:
        val_best = val_loss
        patience = 0
        torch.save(model.state_dict(), filepathModel)
        
    
    ## Early stopping
    if patience == 100:
        break
    patience += 1
    
    '''### Plot test and validation model ###'''
    plt.plot(train_loss_history, label='train')        
    plt.plot(val_loss_history, label='validation')
    plt.title('Loss model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



### Save model
# torch.save(model.state_dict(), FILEPATH) # filepathModel = '/Users/amaliekoch/Desktop/STFT'



'''### Validation model ###'''
#model = ConvNet().to(device)
# model.load_state_dict(torch.load("Filename"))

# model.eval() 
# val_loss_history = list()
# with torch.no_grad():
#     for epoch in range(epochs):
#         val_loss = 0.0
#         for i, (imgs, labels) in enumerate(val_loader):
#             labels = torch.tensor(np.eye(2)[np.asarray(labels)],dtype = torch.float32) #one hot encoding
#             imgs, labels = imgs.to(device), labels.to(device)
#             prediction = model(imgs)
#             lossVal = criterion(prediction, labels)
#             val_loss += lossVal.item()
        
        
#         val_loss = val_loss/len(val_loader)
#         val_loss_history.append(val_loss)
        

'''### Plot test and validation model ###'''
plt.plot(train_loss_history, label='train')        
plt.plot(val_loss_history, label='validation')
plt.title('Loss model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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




