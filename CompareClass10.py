import Validate
from Validate import compareClass10_train
import numpy as np
import matplotlib.pyplot as plt
Train1 = [[90.69, 89.05, 86.07000000000001, 68.81]]
Train2 = [[90.94, 89.91, 87.19, 71.39999999999999]]
Train3 = [[90.84, 90.48, 87.27000000000001, 71.94]]
TrainComposite = [[90.47, 89.53999999999999, 86.02, 70.16]]
TrainTiled = [[92.91, 91.02, 88.53, 78.28]]



TrainCompositeTile3= list()
TrainCompositeTile3.append(compareClass10_train(0.6))

#
# Train1 = list()
# for i in range(0,1):
#     Train1.append(compareClass10_train(0.6,1))
#
# Train2 = list()
# for i in range(0,1):
#     Train2.append(compareClass10_train(0.6,2))

Train3 = list()
for i in range(0, 1):
    Train3.append(compareClass10_train(0.6,3))

Train4 = list()
for i in range(0, 1):
    Train4.append(compareClass10_train(0.6,4))


x = [0, 0.4, 0.6, 0.8]
# plt.plot(x,Baseline[0], color='black', linewidth=2.5,label="Baseline")
plt.plot(x,Train1[0], color='black', linewidth=1.5, label="1 Sets")
plt.plot(x,Train2[0], color='blue', linewidth=1.5, label="2 Sets")
plt.plot(x,Train3[0], color='olive', linewidth=1.5, label="3 Sets")
plt.plot(x,TrainComposite[0], color='cyan', linewidth=1.5, label="Composite Sets")
plt.plot(x,TrainTiled[0], color='red', linewidth=1.5, label="Tiled Set")

plt.legend()
plt.grid()

plt.xlabel('Test Data Corruption Level (p)',fontsize = 13,fontweight='bold')
plt.ylabel('Percent Accuracy (%)',fontsize = 13,fontweight='bold')
plt.title('All 10 Classifier\n N Training Sets Variaton on P=0.6',fontsize = 16,fontweight='bold')




Baseline_Avg = np.nanmean(Baseline,axis=0)
Train1_Avg = np.nanmean(Train1,axis=0)
Train2_Avg = np.nanmean(Train2,axis=0)
Train3_Avg = np.nanmean(Train3,axis=0)
Train4_Avg = np.nanmean(Train4,axis=0)

x = [0, 0.4, 0.6, 0.8]

plt.plot(x,Baseline_Avg, color='black', linewidth=2.5,label="Baseline")
plt.plot(x,Train1_Avg, color='olive', linewidth=1.5, label="1 Sets")
plt.plot(x,Train2_Avg, color='cyan', linewidth=1.5, label="2 Sets")
plt.plot(x,Train3_Avg, color='red', linewidth=1.5, label="3 Sets")
plt.plot(x,Train4_Avg, color='blue', linewidth=1.5, label="4 Sets")

plt.legend()
plt.grid()

plt.xlabel('Test Data Corruption Level (p)',fontsize = 13,fontweight='bold')
plt.ylabel('Percent Accuracy (%)',fontsize = 13,fontweight='bold')
plt.title('All 10 Classifier \n N Training Sets Variaton on P=0.6',fontsize = 16,fontweight='bold')



###########################
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# Baseline = list()
# for i in range(1, 3):
#     Baseline.append(compareClass10_train(0,1))
#
# Corrupt6 = list()
#     try:
#         Corrupt6.append(compareClass10_train(0.6, 2))
#     except:
#         Corrupt6.append(list([np.nan, np.nan, np.nan, np.nan]))
#
#
# Corrupt4 = list()
#     try:
#         Corrupt4.append(compareClass10_train(0.4, 2))
#     except:
#         Corrupt4.append(list([np.nan, np.nan, np.nan, np.nan]))
#
# Corrupt8 = list()
#     try:
#         Corrupt8.append(compareClass10_train(0.8, 2))
#     except:
#         Corrupt8.append(list([np.nan, np.nan, np.nan, np.nan]))
#
#
#
# x = [0, 0.2, 0.4, 0.6]
#
# Baseline_Avg = np.nanmean(Baseline,axis=0)
# # Corrupt2_Avg = np.nanmean(Corrupt2,axis=0)
# Corrupt4_Avg = np.nanmean(Corrupt4,axis=0)
# Corrupt6_Avg = np.nanmean(Corrupt6,axis=0)
# Corrupt8_Avg = np.nanmean(Corrupt8,axis=0)
# # Corrupt2_2_Avg = np.nanmean(Corrupt2_2,axis=0)
# # Corrupt4_2_Avg = np.nanmean(Corrupt4_2,axis=0)
# Corrupt6_2_Avg = np.nanmean(Corrupt6_2,axis=0)
# Corrupt8_2_Avg = np.nanmean(Corrupt8_2,axis=0)
#
#
# plt.plot(x,Baseline_Avg, color='black', linewidth=2.5,label="Baseline")
# # plt.plot(x,Corrupt2_Avg, color='blue', linewidth=1.5, label="Train P=0.2")
# plt.plot(x,Corrupt4, color='red', linewidth=1.5, label="Train P=0.4")
# plt.plot(x,Corrupt6_Avg, color='olive', linewidth=1.5, label="Train P=0.6")
# plt.plot(x,Corrupt8, color='cyan', linewidth=1.5, label="Train P=0.8")
# # plt.plot(x,Corrupt2_2_Avg, color='blue', linewidth=1.5,linestyle='dashed', label="Train P=0.2, N=2")
# # plt.plot(x,Corrupt4_2_Avg, color='red', linewidth=1.5,linestyle='dashed', label="Train P=0.4, N=2")
# plt.plot(x,Corrupt6_2_Avg, color='olive', linewidth=1.5,linestyle='dashed', label="Train P=0.6, N=2")
# plt.legend()
# plt.grid()
#
# plt.xlabel('Test Data Corruption Level (p)',fontsize = 13,fontweight='bold')
# plt.ylabel('Percent Accuracy (%)',fontsize = 13,fontweight='bold')
# plt.title('All 10 Classes\n P Training Variatons',fontsize = 16,fontweight='bold')
