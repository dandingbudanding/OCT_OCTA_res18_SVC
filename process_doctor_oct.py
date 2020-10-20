# -*- coding: utf-8 -*-
import pandas as pd                         #导入pandas包
data = pd.read_csv("OCT.csv")           	#读取csv文件

data0=data[data['D']==0]
data1=data[data['D']==1]

row0=data0.shape[0]
row1=data1.shape[0]

#average
count0=data0[data0==0].count()
count00=count0['A']+count0['B']+count0['C']
count1=data1[data1==1].count()
count11=count1['A']+count1['B']+count1['C']

percentage0=count00*1.0/(row0*3.0)
percentage1=count11*1.0/(row1*3.0)

print(percentage0,percentage1)


# each
count_doct1=[count0['A'],count1['A']]
count_doct2=[count0['B'],count1['B']]
count_doct3=[count0['C'],count1['C']]

chushu=[count0['D'],count1['D']]

import numpy as np
percentage0_doct1=np.array(count_doct1)/np.array(chushu)
percentage1_doct2=np.array(count_doct2)/np.array(chushu)
percentage2_doct3=np.array(count_doct3)/np.array(chushu)

print(percentage0_doct1,percentage1_doct2,percentage2_doct3)

# 1:0.98:1:0.92
# 0.8333333333333334 0.7133333333333334 0.86 0.94

#draw precission bar pic avg

import matplotlib.pyplot as plt

name_list = ["inactive CNV", "active CNV"]
num_list_model = [0.764704, 0.666664]
num_list_doctor = [0.8571428571428571, 0.8823529411764706]
x = list(range(len(num_list_model)))
total_width, n = 0.4, 2
width = total_width / n


plt.bar(x, num_list_model, width=width, label='AM-F', fc='lightcoral')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor, width=width, label='Doctor', tick_label=name_list, fc='peru')
plt.legend(loc="upper left",bbox_to_anchor=(0.0, 1.1),ncol=2)
plt.savefig('./savedimg/OCT/CNN_Doctor_precission_comparision_AVG.png', dpi=300)
plt.show()

#draw precission bar pic each

import matplotlib.pyplot as plt

name_list = ["inactive CNV", "active CNV"]
num_list_model = [0.764704, 0.666664]
num_list_doctor1 = [0.77142857, 0.82352941]
num_list_doctor2 = [0.82857143, 1.]
num_list_doctor3 = [0.97142857, 0.82352941]
x = list(range(len(num_list_model)))
total_width, n = 0.4, 2
width = total_width / n


plt.bar(x, num_list_model, width=width, label='AM-F', fc='lightcoral')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor, width=width, label='Doctor1', tick_label=name_list, fc='peru')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor2, width=width, label='Doctor2', tick_label=name_list, fc='greenyellow')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor3, width=width, label='Doctor3', tick_label=name_list, fc='plum')

plt.legend(loc="upper left",bbox_to_anchor=(0.0, 1.1),ncol=4)
plt.savefig('./savedimg/OCT/CNN_Doctor_precission_comparision_EACH.png', dpi=300)
plt.show()


#draw recall bar pic each
from metrics import calc_metrics,cal_confu_matrix,metrics
import numpy as np

aaaaaa=np.array(data['A'])
confu_matrix=cal_confu_matrix(np.array(data['A']),np.array(data['D']),class_num=4)
print(confu_matrix)
metrics(confu_matrix, save_path="./savedimg/OCT/doctor1")

confu_matrix=cal_confu_matrix(np.array(data['B']),np.array(data['D']),class_num=4)
print(confu_matrix)
metrics(confu_matrix, save_path="./savedimg/OCT/doctor2")

confu_matrix=cal_confu_matrix(np.array(data['C']),np.array(data['D']),class_num=4)
print(confu_matrix)
metrics(confu_matrix, save_path="./savedimg/OCT/doctor3")

name_list = ["inactive CNV", "active CNV"]
num_list_model = [0.866664,	0.500000]
num_list_doctor1 = [0.90, 0.63636]
num_list_doctor2 = [1.0, 0.73913]
num_list_doctor3 = [0.91892, 0.93333]

print((np.array(num_list_doctor1)+np.array(num_list_doctor2)+np.array(num_list_doctor3))/3.0)
x = list(range(len(num_list_model)))
total_width, n = 0.4, 2
width = total_width / n

plt.bar(x, num_list_model, width=width, label='AM-F', fc='lightcoral')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor1, width=width, label='Doctor1', tick_label=name_list, fc='peru')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor2, width=width, label='Doctor2', tick_label=name_list, fc='greenyellow')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor3, width=width, label='Doctor3', tick_label=name_list, fc='plum')

plt.legend(loc="upper left",bbox_to_anchor=(0.0, 1.1),ncol=4)
plt.savefig('./savedimg/OCT/CNN_Doctor_recall_comparision_EACH.png', dpi=300)
plt.show()



#draw recall bar pic avg
name_list = ["inactive CNV", "active CNV"]
num_list_model = [0.866664,	0.500000]
num_list_doctor = [0.93332767, 0.82779667]

x = list(range(len(num_list_model)))
total_width, n = 0.4, 2
width = total_width / n

plt.bar(x, num_list_model, width=width, label='AM-F', fc='lightcoral')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list_doctor, width=width, label='Doctor', tick_label=name_list, fc='peru')

plt.legend(loc="upper left",bbox_to_anchor=(0.0, 1.1),ncol=2)
plt.savefig('./savedimg/OCT/CNN_Doctor_recall_comparision_AVG.png', dpi=300)
plt.show()

# # PRE
# # EACH:
# num_list_model = [0.764704, 0.666664]
# num_list_doctor1 = [0.77142857, 0.82352941]
# num_list_doctor2 = [0.82857143, 1.]
# num_list_doctor3 = [0.97142857, 0.82352941]
# # AVG:
# num_list_model = [0.764704, 0.666664]
# num_list_doctor = [0.8571428571428571, 0.8823529411764706]
# #
# # RECALL
# # EACH:
# num_list_model = [0.866664,	0.500000]
# num_list_doctor1 = [0.90, 0.63636]
# num_list_doctor2 = [1.0, 0.73913]
# num_list_doctor3 = [0.91892, 0.93333]
# # AVG:
# num_list_model = [0.866664,	0.500000]
# num_list_doctor = [0.93332767, 0.82779667]