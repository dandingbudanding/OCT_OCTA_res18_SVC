import os
import numpy as np
import argparse
import torch
import torch.optim as optim
from tqdm import *
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import copy
import time

from model.res18model import res18,res18_oct_or_octa
from mydataloader import trainset,testset



trainloader = torch.utils.data.DataLoader(trainset(), batch_size=24,
                                              shuffle=True)
testloader = torch.utils.data.DataLoader(testset(), batch_size=24,
                                              shuffle=True)

# ######################################################################
# # Define the Embedding Network
#
# class ClassificationNetwork(nn.Module):
#     def __init__(self):
#         super(ClassificationNetwork, self).__init__()
#         self.convnet = torchvision.models.resnet18(pretrained=False)
#         num_ftrs = self.convnet.fc.in_features
#         self.convnet.fc = nn.Linear(num_ftrs, 64)
#
#     def forward(self, inputs):
#         outputs = self.convnet(inputs)
#
#         return outputs
#
# class EmbeddingNetwork(nn.Module):
#     def __init__(self):
#         super(EmbeddingNetwork, self).__init__()
#         self.resnet = ClassificationNetwork()
#         self.cls = self.resnet.convnet.fc
#
#         self.conv1 = nn.Conv2d(6,64,3)#self.resnet.convnet.conv1
#         self.bn1 = self.resnet.convnet.bn1
#         self.relu = self.resnet.convnet.relu
#         self.maxpool = self.resnet.convnet.maxpool
#         self.layer1 = self.resnet.convnet.layer1
#         self.layer2 = self.resnet.convnet.layer2
#         self.layer3 = self.resnet.convnet.layer3
#         self.layer4 = self.resnet.convnet.layer4
#         self.layer4 = self.resnet.convnet.layer4
#         self.avgpool = self.resnet.convnet.avgpool
#
#     def forward(self, x1,x2):
#         x=torch.cat([x1,x2],1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         layer1 = self.layer1(x)  # (, 64L, 56L, 56L)
#         layer2 = self.layer2(layer1)  # (, 128L, 28L, 28L)
#         layer3 = self.layer3(layer2)  # (, 256L, 14L, 14L)
#         layer4 = self.layer4(layer3)  # (,512,7,7)
#         x = self.avgpool(layer4)  # (,512,1,1)
#
#
#         x = x.view(x.size(0), -1)
#         return x

classificationNetwork = res18_oct_or_octa().cuda()

#############################################
# Define the optimizer

criterion = nn.CrossEntropyLoss()

optimizer_embedding = optim.Adam([
    {'params': classificationNetwork.parameters()},
], lr=0.001)

embedding_lr_scheduler = lr_scheduler.StepLR(optimizer_embedding, step_size=10, gamma=0.5)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^


def train_model(model, criterion, optimizer, scheduler, num_epochs=400):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        running_loss = 0.0
        tot_dist = 0.0
        running_corrects = 0
        loss = 0

        # Iterate over data.
        COUNT=0
        for i, (OCT,OCTA, labels) in tqdm(enumerate(trainloader)):
            label = labels
            # wrap them in Variable
            OCTA = Variable(OCTA.cuda())
            labels = Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(OCTA)
            _, preds = torch.max(outputs.data.cpu(), 1)

            labels = labels.view(labels.size(0))

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            aaa = loss.data.item()
            # running_loss += loss.data[0] * inputs.size(0)
            running_loss += aaa
            label_=label.unsqueeze(1)
            preds_=preds.unsqueeze(1)
            running_corrects += torch.sum(preds_ == label_)
            COUNT+=1

        epoch_loss = running_loss / (COUNT * 1.0)
        epoch_acc = running_corrects / (COUNT * 1.0)


        print('{} Loss: {:.4f} Train Accuracy: {:.4f} /n'.format(epoch, epoch_loss, epoch_acc))

        # deep copy the model
        if  epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 30 == 0:
            torch.save(best_model_wts, os.path.join('./' + 'res18_octa.pth'))
            print('save!')


    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


classificationNetwork = train_model(classificationNetwork, criterion, optimizer_embedding,
                                    embedding_lr_scheduler, num_epochs=1000)

torch.save(classificationNetwork.state_dict(), os.path.join(r'./' + 'res18_octa.pth'))


