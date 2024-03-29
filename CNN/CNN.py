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
import matplotlib.pyplot as plt
import mpu
import plyer as pl
import torchvision
import numpy as np


'''### Device configuration ###'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 
'''### Hyperparameters ###'''

'''### Data ###'''
def torch_loader(path):
    sample = torch.load(path)
    return sample

# # Writer will output to ./runs/ directory by default
batch_size = 6
minibatch = 250
epochs = 200
learning_rate = 0.0001
patientswait = 30
ch1 = 12
ch2 = 8
ch3 = 10

fc1in = 96
fciout = 10 
#path = pl.filechooser.choose_dir()
path = r'C:\Users\clark\Desktop\STFT\train'
train_data = torchvision.datasets.DatasetFolder(root=path,
                                                loader=torch_loader,
                                                extensions=['.pt'],
                                                )
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#path = pl.filechooser.choose_dir()
path = r'C:\Users\clark\Desktop\STFT\test' 
val_data = torchvision.datasets.DatasetFolder(root=path,
                                                loader=torch_loader,
                                                extensions=['.pt']
                                                )

val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

'''### Model definition ###'''
### Define architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=ch1, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=ch2, out_channels=ch3, kernel_size=2, padding=1)
        #self.conv4 = nn.Conv2d(in_channels=ch3, out_channels=ch4, kernel_size=5, padding=1)
        #self.conv5 = nn.Conv2d(in_channels=ch4, out_channels=ch5, kernel_size=5, padding=1)

        self.conv2_drop = nn.Dropout2d()
        #self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 1) #Sikre at vi har et lige antal efter første pooling
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=3) #Den normale pooling vi havde fra starten
        self.fc1 = nn.Linear(in_features=fc1in, out_features=fciout)
        self.fc2 = nn.Linear(in_features=fciout, out_features=2)

        self.activation = torch.nn.Softmax(dim=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.leaky = torch.nn.LeakyReLU()
        
        
    def forward(self, x):
        x = self.pool2(self.leaky(self.conv1(x)))
        x = self.pool2(self.leaky(self.conv2_drop(self.conv2(x)))) 
        #x = self.pool2(F.relu(self.conv2_drop(self.conv3(x)))) 
        #x = self.pool2(self.leaky(self.conv2_drop(self.conv4(x)))) 
        #x = self.pool2(self.leaky(self.conv3(x)))
        #print(torch.Tensor.size(x))
        x = x.view(-1, fc1in)
        x= F.dropout(x, p=0.8, training=self.training)        
        x = self.sigmoid1(self.fc1(x))    
        x= F.dropout(x, p=0.8, training=self.training)           
        x = self.sigmoid1(self.fc2(x))                            
        x = self.activation(x)
        return x
    
### Instantiate the network
model = ConvNet().to(device)
### Define the optimizer
optimizer = optim.NAdam(model.parameters(),lr=learning_rate)
### Define the loss function
criterion = nn.BCELoss()

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
            # if i % minibatch == minibatch-1:  # print every 40 batches
            #     print('[Epoch: {} Batch: {}/{}] loss: {}'.format(
            #         epoch + 1, 
            #         i + 1, 
            #         len(val_loader), 
            #         val_loss / (i*batch_size))) #fix denne algoritme, den virker ikke korrekt
    ### Save loss in history        
    train_loss = train_loss/len(train_loader)
    train_loss_history.append(train_loss)
    val_loss = val_loss/len(val_loader)
    val_loss_history.append(val_loss)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
          
    ## Model chechpoint
    if val_loss < val_best:
        #torch.save(model.state.dict(), filepath)
        val_best = val_loss
        print('Val loss: '+ str(val_best))
        patience = 0
    
    ## Early stopping
    if patience == patientswait:
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

    #writer.add_scalars('Loss', {'Training Loss': train_loss,'Validation Loss': val_loss,}, epoch)
    #writer.add_scalar('Validation Accuracy', acc,epoch)
### Save model
# torch.save(model.state_dict(), FILEPATH)



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