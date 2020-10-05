from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import cv2
import numpy as np

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
# 02867311_Min_Jian__1182_Angio_Retina_OD_2018-12-10_15-53-04_M_1955-12-19_Enface-304x304.png
def default_loader(path):
    img_pil = cv2.imread(path,1).astype(np.float32)/255.0
    img_pil = cv2.resize(img_pil,(224,224))
    img_tensor = preprocess(img_pil)
    return img_tensor

#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, root=r"F://DATA//OCTA+OCT_DATA//train//",loader=default_loader):
        self.folder=os.listdir(root)
        self.folder0=os.listdir(os.path.join(root,self.folder[0]))
        self.folder1 = os.listdir(os.path.join(root,self.folder[1]))
        n0 = len(os.listdir(os.path.join(root,self.folder[0],self.folder0[1])))
        n1 = len(os.listdir(os.path.join(root,self.folder[1],self.folder1[1])))

        self.MASKROOT = os.path.join(root, self.folder[0], self.folder0[0])
        self.OCTROOT=os.path.join(root,self.folder[0],self.folder0[1])
        self.OCTAROOT = os.path.join(root, self.folder[0], self.folder0[2])

        self.loader = loader

        self.OCT_imgs = []
        self.OCTA_imgs = []
        self.labels=[]
        for file in os.listdir(self.OCTROOT):
            self.OCT_imgs.append(self.loader(os.path.join(self.OCTROOT, file)))
            octa=cv2.imread(os.path.join(self.OCTAROOT, file),1)
            mask=cv2.imread(os.path.join(self.MASKROOT, file),1)
            octa=cv2.bitwise_and(octa,mask)
            octa = cv2.resize(octa, (224, 224)).astype(np.float32)/255.0
            octa = preprocess(octa)
            self.OCTA_imgs.append(octa)
            self.labels.append(0)

        self.MASKROOT = os.path.join(root, self.folder[1], self.folder0[0])
        self.OCTROOT = os.path.join(root, self.folder[1], self.folder0[1])
        self.OCTAROOT = os.path.join(root, self.folder[1], self.folder0[2])
        for file in os.listdir(self.OCTROOT):
            self.OCT_imgs.append(self.loader(os.path.join(self.OCTROOT, file)))
            octa = cv2.imread(os.path.join(self.OCTAROOT, file), 1)
            mask = cv2.imread(os.path.join(self.MASKROOT, file), 1)
            octa = cv2.bitwise_and(octa, mask)
            octa = cv2.resize(octa, (224, 224)).astype(np.float32)/255.0
            octa = preprocess(octa)
            self.OCTA_imgs.append(octa)
            self.labels.append(1)



    def __getitem__(self, index):
        OCT_img = self.OCT_imgs[index]
        OCTA_img = self.OCTA_imgs[index]
        label = self.labels[index]
        return OCT_img,OCTA_img,label

    def __len__(self):
        return len(self.OCT_imgs)


class testset(Dataset):
    def __init__(self, root=r"F://DATA//OCTA+OCT_DATA//test//",loader=default_loader):
        self.folder = os.listdir(root)
        self.folder0 = os.listdir(os.path.join(root, self.folder[0]))
        self.folder1 = os.listdir(os.path.join(root, self.folder[1]))
        n0 = len(os.listdir(os.path.join(root, self.folder[0], self.folder0[1])))
        n1 = len(os.listdir(os.path.join(root, self.folder[1], self.folder1[1])))

        self.MASKROOT = os.path.join(root, self.folder[0], self.folder0[0])
        self.OCTROOT = os.path.join(root, self.folder[0], self.folder0[1])
        self.OCTAROOT = os.path.join(root, self.folder[0], self.folder0[2])

        self.loader = loader

        self.OCT_imgs = []
        self.OCTA_imgs = []
        self.labels = []
        for file in os.listdir(self.OCTROOT):
            self.OCT_imgs.append(self.loader(os.path.join(self.OCTROOT, file)))
            octa = cv2.imread(os.path.join(self.OCTAROOT, file), 1)
            mask = cv2.imread(os.path.join(self.MASKROOT, file), 1)
            octa = cv2.bitwise_and(octa, mask)
            octa = cv2.resize(octa, (224, 224)).astype(np.float32)/255.0
            octa = preprocess(octa)
            self.OCTA_imgs.append(octa)
            self.labels.append(0)

        self.MASKROOT = os.path.join(root, self.folder[1], self.folder0[0])
        self.OCTROOT = os.path.join(root, self.folder[1], self.folder0[1])
        self.OCTAROOT = os.path.join(root, self.folder[1], self.folder0[2])
        for file in os.listdir(self.OCTROOT):
            self.OCT_imgs.append(self.loader(os.path.join(self.OCTROOT, file)))
            octa = cv2.imread(os.path.join(self.OCTAROOT, file), 1)
            mask = cv2.imread(os.path.join(self.MASKROOT, file), 1)
            octa = cv2.bitwise_and(octa, mask)
            octa = cv2.resize(octa, (224, 224)).astype(np.float32)/255.0
            octa = preprocess(octa)
            self.OCTA_imgs.append(octa)
            self.labels.append(1)



    def __getitem__(self, index):
        OCT_img = self.OCT_imgs[index]
        OCTA_img = self.OCTA_imgs[index]
        label = self.labels[index]
        return OCT_img,OCTA_img,label

    def __len__(self):
        return len(self.OCT_imgs)

# aaa=trainset()
# bbb=testset()