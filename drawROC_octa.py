# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


import torch
from torch.autograd import Variable
import os
from torch import nn
from scipy import interp

gpu_id="0,1" ; #指定gpu id
#配置环境  也可以在运行时临时指定 CUDA_VISIBLE_DEVICES='2,7' Python train.py
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id #这里的赋值必须是字符串，list会报错
device_ids=range(torch.cuda.device_count())  #torch.cuda.device_count()=2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model.res18model import res18,res18_oct_or_octa
from mydataloader import trainset, testset


# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
# 将标签二值化
y = label_binarize(y, classes=[0, 1])
# 设置种类
n_classes = y.shape[1]
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
        #180 0.8116  210 0.7729 240 0.8043  270:0.837 300:0.8442   360:0.8490 390:0.8297  420:0.8418 450:0.8297

        #60 0.8176  120:0.8249      810:0.8285  720:0.9034
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
# net = reload_net()
# model_weights = torch.load("./res34finetune/net_params_49.pkl")
# model_weights = torch.load("./res34alldata/net_params_24_all.pkl")
model = EmbeddingNetwork().to(device)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
from sklearn.svm import SVC

fp = open("octa.pkl","rb+")
import pickle
SVCmodel=pickle.load(fp)#序列化打印结果
fp.close()


resultall = []
labelall=[]
lableforconfusion=[]
for i, data in enumerate(testloader, 0):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
    # enumerate是python的内置函数，既获得索引也获得数据
    # get the inputs
    oct, octa, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
    oct, octa, labels = oct.to(device), octa.to(device), labels.to(device)
    outputs_fromCNN = model(Variable(oct))
    outputs=SVCmodel.predict_proba(outputs_fromCNN.cpu().detach().numpy())

    lableforconfusion.extend(labels.cpu().numpy())
    class_num = 2
    ones = torch.eye(class_num)
    labels_=ones.index_select(0, labels.cpu())

    # ones = torch.eye(class_num)
    # outputs = ones.index_select(0, torch.from_numpy(outputs))

    resultall.extend(outputs)
    labelall.extend(labels_.numpy())

resultall=np.array(resultall)
labelall=np.array(labelall)
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

y_test=labelall
y_score=resultall

from metrics import calc_metrics,cal_confu_matrix,metrics
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap="summer")    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,rotation=90)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
predicted = np.argmax(resultall, 1)
confu_matrix = cal_confu_matrix(np.array(predicted), np.array(lableforconfusion), class_num=2)
print(confu_matrix)

labels_name = ["inactive CNV", "active CNV"]
plot_confusion_matrix(confu_matrix, labels_name, "Confusion Matrix")
plt.savefig('./savedimg/confusionmatrix_octa.png', dpi=300, bbox_inches='tight')
plt.show()
metrics(confu_matrix, save_path="./savedimg/")

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）    micro和macro:https://www.cnblogs.com/techengin/p/8962024.html
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['LawnGreen','aqua'])
labels_name = ["inactive CNV", "active CNV"]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(labels_name[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves')
plt.legend(loc="lower right")

plt.savefig('./savedimg/ROC_octa.png', dpi=300)
plt.show()