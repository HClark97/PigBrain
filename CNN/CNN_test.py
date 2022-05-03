# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:54:08 2022

@author: clark
"""

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

'''### Device configuration ###'''
device = torch.device('cpu')
# 
'''### Hyperparameters ###'''

'''### Data ###'''
def torch_loader(path):
    sample = torch.load(path)
    return sample

# # Writer will output to ./runs/ directory by default
batch_size = 50
minibatch = 250
epochs = 200
learning_rate = 0.0001
patientswait = 30
ch1 = 4
ch2 = 8
ch3 = 10
fc1in = 96
fciout = 10 

all_y_true = []
all_y_pred = []

path = r'C:\Users\clark\Desktop\STFT\Test'
test_data = torchvision.datasets.DatasetFolder(root=path,
                                                loader=torch_loader,
                                                extensions=['.pt'],
                                                )

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

'''### Model definition ###'''
### Define architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=0) 
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride = 3) #Sikre at vi har et lige antal efter første pooling
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) #Den normale pooling vi havde fra starten
        self.fc1 = nn.Linear(in_features=120, out_features=30)
        self.fc2 = nn.Linear(in_features=30, out_features=2)
        self.activation = torch.nn.Softmax(dim=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.leaky = torch.nn.LeakyReLU()
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2_drop(self.conv2(x)))) 
        x = x.view(-1, 120)
        x= F.dropout(x, p=0.7, training=self.training)        
        x = self.sigmoid1(self.fc1(x))    
        x= F.dropout(x, p=0.7, training=self.training)           
        x = self.sigmoid1(self.fc2(x))                            
        x = self.activation(x)
        return x



'''### Testing model ###'''

model = ConvNet().to(device)
model.load_state_dict(torch.load(r"C:\Users\clark\Desktop\STFT\model4.pth"))
model.eval()
n_correct = 0
n_samples = 0
y_true = []
y_pred = []
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        temp_label = labels.numpy()
        y_true.extend(temp_label)
        labels = torch.tensor(np.eye(2)[np.asarray(labels)],dtype = torch.float32) #one hot encoding, so we got a 32,2 matrix (Alex said this is how it is done)
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        prob, predicted = torch.max(outputs, 1)
        temp_pred = outputs[:,1]
        temp_pred = temp_pred.numpy()
        y_pred.extend(temp_pred)
        predicted = torch.tensor(np.eye(2)[np.asarray(predicted)],dtype = torch.float32)
        n_samples += labels.size(0)
        n_correct += (torch.round(predicted[:,0]) == labels[:,0]).sum().item()
        
        
accuracy = 100*n_correct/n_samples
print(f'Accuracy of the network: {accuracy} %')


# Calculate image-level ROC AUC score
fpr, tpr, threshold = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

plt.figure(1)
lw = 1
plt.plot(fpr, tpr, color="darkorange", label="CNN (area = {:.3f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()