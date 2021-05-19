from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from google.colab import drive
#drive.mount('/content/MyDrive', force_remount=True)
#import os
#os.chdir("MyDrive/MyDrive/assignment5_materials/assignment5_materials")
data_dir="face_mask_dataset"
num_classes=2
batch_size=128
num_epochs=20
model_ft= models.mobilenet_v3_large(pretrained=True);
for param in model_ft.parameters():
    param.requires_grad= False
num_ftrs= model_ft.classifier[3].in_features
model_ft.classifier[3]= nn.Linear(num_ftrs,num_classes)
input_size=224 

data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders")

image_datasets= {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Validation']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['Train', 'Validation']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft=model_ft.to(device)
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
best_model_wts = copy.deepcopy(model_ft.state_dict())
best_acc = 0.0
for epoch in range(num_epochs):
    for phase in ["Train","Validation"]:
        if phase== "Train":
            model_ft.train()
        else:
            model_ft.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders_dict[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()
            #print(inputs)    
            with torch.set_grad_enabled(phase=="Train"):
                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                if phase== "Train":
                    #print(loss)
                    loss.backward()
                    optimizer_ft.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
    if phase == 'Validation' and epoch_acc > best_acc:
      best_acc = epoch_acc
      best_model_wts = copy.deepcopy(model_ft.state_dict())
    print(epoch,phase+":",epoch_loss,epoch_acc.item())
model_ft.load_state_dict(best_model_wts)
torch.save(model_ft,"MyDrive/MyDrive/mask_classifier.pth")
