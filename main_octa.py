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
import numpy

import getpass

from model.res18model import res18,res18_oct_or_octa
from mydataloader import trainset, testset

trainloader = torch.utils.data.DataLoader(trainset(), batch_size=48,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset(), batch_size=48,
                                         shuffle=True)


######################################################################
# Define the Embedding Network

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = res18

        num_ftrs = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(num_ftrs, 64)

    def forward(self, inputs):
        outputs = self.convnet(inputs)
        return outputs


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.resnet = res18_oct_or_octa()
        self.resnet.load_state_dict(torch.load('./res18_octa.pth'))
        self.cls = self.resnet.cls
        self.cls.load_state_dict(self.resnet.cls.state_dict())

        self.conv1 = self.resnet.conv1
        self.conv1.load_state_dict(self.resnet.conv1.state_dict())
        self.bn1 = self.resnet.bn1
        self.bn1.load_state_dict(self.resnet.bn1.state_dict())
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer1.load_state_dict(self.resnet.layer1.state_dict())
        self.layer2 = self.resnet.layer2
        self.layer2.load_state_dict(self.resnet.layer2.state_dict())
        self.layer3 = self.resnet.layer3
        self.layer3.load_state_dict(self.resnet.layer3.state_dict())
        self.layer4 = self.resnet.layer4
        self.layer4.load_state_dict(self.resnet.layer4.state_dict())
        self.avgpool = self.resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x)  # (, 64L, 56L, 56L)
        layer2 = self.layer2(layer1)  # (, 128L, 28L, 28L)
        layer3 = self.layer3(layer2)  # (, 256L, 14L, 14L)
        layer4 = self.layer4(layer3)  # (,512,7,7)
        x = self.avgpool(layer4)  # (,512,1,1)

        x = x.view(x.size(0), -1)
        return x


embedding = EmbeddingNetwork().cuda()
#############################################
# Test the Embedding network


#############################################
# Define the optimizer


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
from sklearn.svm import SVC

def Evaluate(num_epochs):
    since = time.time()
    # parameter should be set according to the performance on validation set
    model = SVC(C=10,probability = True)
    # for simplicity only 1 epoch ,delete the part of calculate confidence interval
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        running_loss = 0.0
        running_corrects = 0.0
        total = 0

        # Iterate over data.

        embedding.train(False)

        for _,(OCT_train, OCTA_train, labels_train) in tqdm(enumerate(trainloader)):
            # wrap them in Variable
            OCTA_train = Variable(OCTA_train.cuda())
            labels_train = Variable(labels_train.cuda())

            support_feature = embedding(OCTA_train).data.cpu()


            support_feature = torch.squeeze(support_feature, 0).numpy()
            support_belong = torch.squeeze(labels_train.cpu(), 0).numpy()



            support_belong = support_belong.ravel()


            model.fit(support_feature, support_belong)
            for __, (OCT_test, OCTA_test, labels_test) in tqdm(enumerate(testloader)):
                OCTA_test = Variable(OCTA_test.cuda())
                labels_test = Variable(labels_test.cuda())

                test_feature = embedding(OCTA_test).data.cpu()
                test_feature = torch.squeeze(test_feature, 0).numpy()
                test_belong = torch.squeeze(labels_test.cpu(), 0).numpy()

                test_belong = test_belong.ravel()

                Ans = model.predict(test_feature)  # array
                Ans = numpy.array(Ans)

                running_corrects += (Ans == test_belong).sum()
                total += test_feature.shape[0]

            Accuracy = running_corrects / (total * 1.0)
            info = {
                'Accuracy': Accuracy,
            }

    print('Accuracy: {:.4f} '.format(Accuracy))
    import pickle
    with open("octa.pkl","wb") as f:
        pickle.dump(model,f)

Evaluate(num_epochs=20)
# Accuracy: 0.8086
