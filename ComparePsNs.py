import Validate
from Validate import validate_1and7
from Validate import CorruptTrain_1and7

import numpy as np
import matplotlib.pyplot as plt

Baseline = list()
for i in range(1,11):
    Baseline.append(validate_1and7())

Corrupt2 = list()
Corrupt2_2 = list()
for i in range(1,11):
    try:
        Corrupt2.append(CorruptTrain_1and7(0.2,1))
    except:
        Corrupt2.append(list([np.nan,np.nan,np.nan,np.nan]))

    try:
        Corrupt2_2.append(CorruptTrain_1and7(0.2,2))
    except:
        Corrupt2_2.append(list([np.nan,np.nan,np.nan,np.nan]))


Corrupt4 = list()
Corrupt4_2 = list()
for i in range(1,11):
    try:
        Corrupt4.append(CorruptTrain_1and7(0.4,1))
    except:
        Corrupt4.append(list([np.nan,np.nan,np.nan,np.nan]))

    try:
        Corrupt4_2.append(CorruptTrain_1and7(0.4,2))
    except:
        Corrupt4_2.append(list([np.nan,np.nan,np.nan,np.nan]))

Corrupt6 = list()
Corrupt6_2 = list()
for i in range(1,11):
    try:
        Corrupt6.append(CorruptTrain_1and7(0.6,1))
    except:
        Corrupt6.append(list([np.nan,np.nan,np.nan,np.nan]))

    try:
        Corrupt6_2.append(CorruptTrain_1and7(0.6,2))
    except:
        Corrupt6_2.append(list([np.nan,np.nan,np.nan,np.nan]))

Corrupt8 = list()
Corrupt8_2 = list()
for i in range(1,11):
    try:
        Corrupt8.append(CorruptTrain_1and7(0.8,1))
    except:
        Corrupt8.append(list([np.nan,np.nan,np.nan,np.nan]))

    try:
        Corrupt8_2.append(CorruptTrain_1and7(0.,2))
    except:
        Corrupt8_2.append(list([np.nan,np.nan,np.nan,np.nan]))


x = [0, 0.2, 0.4, 0.6]

Baseline_Avg = np.nanmean(Baseline,axis=0)
Corrupt2_Avg = np.nanmean(Corrupt2,axis=0)
Corrupt4_Avg = np.nanmean(Corrupt4,axis=0)
Corrupt6_Avg = np.nanmean(Corrupt6,axis=0)
Corrupt8_Avg = np.nanmean(Corrupt8,axis=0)
Corrupt2_2_Avg = np.nanmean(Corrupt2_2,axis=0)
Corrupt4_2_Avg = np.nanmean(Corrupt4_2,axis=0)
Corrupt6_2_Avg = np.nanmean(Corrupt6_2,axis=0)


plt.plot(x,Baseline_Avg, color='black', linewidth=2.5,label="Baseline")
plt.plot(x,Corrupt2_Avg, color='blue', linewidth=1.5, label="Train P=0.2")
plt.plot(x,Corrupt4_Avg, color='red', linewidth=1.5, label="Train P=0.4")
plt.plot(x,Corrupt6_Avg, color='olive', linewidth=1.5, label="Train P=0.6")
plt.plot(x,Corrupt8_Avg, color='cyan', linewidth=1.5, label="Train P=0.8")
plt.plot(x,Corrupt2_2_Avg, color='blue', linewidth=1.5,linestyle='dashed', label="Train P=0.2, N=2")
plt.plot(x,Corrupt4_2_Avg, color='red', linewidth=1.5,linestyle='dashed', label="Train P=0.4, N=2")
plt.plot(x,Corrupt6_2_Avg, color='olive', linewidth=1.5,linestyle='dashed', label="Train P=0.6, N=2")
plt.legend()
plt.grid()

plt.xlabel('Test Data Corruption Level (p)',fontsize = 13,fontweight='bold')
plt.ylabel('Percent Accuracy (%)',fontsize = 13,fontweight='bold')
plt.title('Single Classifier 1 and 7\n P and N Training Variatons',fontsize = 16,fontweight='bold')
