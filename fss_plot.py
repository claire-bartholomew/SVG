import iris
import matplotlib.pyplot as plt
import numpy as np
import pdb

#timestep = range(0, 60, 5)
timestep = range(0, 75, 15)

p_25_1_0p1 = [1, 0.87, 0.73, 0.61, 0.52] #t+30 and T+60 need updating
enn_25_1_0p1 = [1, 0.85, 0.74, 0.66, 0.6]#t+30 and T+60 need updating
on_25_1_0p1 = [1, 0.92, 0.86, 0.78, 0.73]#t+30 and T+60 need updating

p_25_4_0p1 = [1, 0.76, 0.5, 0.39, 0.30]#t+30 needs updating
enn_25_4_0p1 = [1, 0.66, 0.51, 0.38, 0.31]#t+30 needs updating
on_25_4_0p1 = [1, 0.83, 0.7, 0.6, 0.52]#t+30 needs updating
## Plot timeseries
plt.plot(timestep, p_25_4_0p1, color='orange', linestyle='-.', label='Persistence')
#plt.plot(timestep, p_36_4, color='orange', linestyle='--', label='Per n=36')
#plt.plot(timestep, p_81_4, color='orange', linestyle='-', label='Per n=81')
plt.plot(timestep, enn_25_4_0p1, color='blue', linestyle='-.', label='Ensemble NN')
#plt.plot(timestep, enn_36_4, color='blue', linestyle='--', label='Ens n=36')
#plt.plot(timestep, enn_81_4, color='blue', linestyle='-', label='Ens n=81')
plt.plot(timestep, on_25_4_0p1, color='green', linestyle='-.', label='Op nowcast')
#plt.plot(timestep, nn_36_4, color='green', linestyle='--', label='Det n=36')
#plt.plot(timestep, nn_81_4, color='green', linestyle='-', label='Det n=81')
#plt.plot(on_timestep, on_9_4, color='red', linestyle='-.', label='Op n=9')
#plt.plot(on_timestep, on_81_4, color='red', linestyle='-', label='Op n=81')

plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.title('4 mm/hr threshold') #, 9x9 neighbourhood')
plt.legend(fontsize=10, ncol=3)
plt.show()
plt.close()

pdb.set_trace()


p_81_1 = [1, 0.98, 0.97, 0.94, 0.93, 0.90, 0.87, 0.84, 0.82, 0.80, 0.78, 0.76]
nn_81_1 = [1, 0.95, 0.93, 0.91, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.77, 0.75, 0.73]
enn_81_1 = [1, 0.96, 0.93, 0.91, 0.88, 0.86, 0.83, 0.81, 0.79, 0.77, 0.75, 0.74]
on_81_1 = []

p_9_1 = [1, 0.94, 0.89, 0.84, 0.80, 0.77, 0.73, 0.71, 0.69, 0.67, 0.65, 0.63]
nn_9_1 = [1, 0.91, 0.85, 0.81, 0.77, 0.73, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]
enn_9_1 = [1, 0.91, 0.85, 0.81, 0.77, 0.74, 0.71, 0.69, 0.68, 0.66, 0.64, 0.63]
on_9_1 = []

p_81_4 = [1, 0.95, 0.88, 0.81, 0.74, 0.64, 0.56, 0.50, 0.43, 0.38, 0.33, 0.29]
enn_81_4 = [1, 0.93, 0.87, 0.81, 0.77, 0.72, 0.65, 0.63, 0.61, 0.56, 0.53, 0.53]
nn_81_4 = [1, 0.91, 0.85, 0.83, 0.76, 0.70, 0.66, 0.64, 0.60, 0.58, 0.57, 0.53]
on_81_4 = [1, 0.71, 0.64, 0.58, 0.50] #15 min timesteps

p_36_4 = [1, 0.92, 0.82, 0.72, 0.63, 0.52, 0.45, 0.39, 0.33, 0.29, 0.25, 0.22]
enn_36_4 = [1, 0.90, 0.83, 0.75, 0.71, 0.65, 0.59, 0.57, 0.55, 0.50, 0.47, 0.46]
nn_36_4 = []

p_9_4 = [1, 0.83, 0.66, 0.53, 0.45, 0.37, 0.32, 0.28, 0.24, 0.20, 0.18, 0.15]
enn_9_4 = [1, 0.81, 0.71, 0.61, 0.57, 0.53, 0.47, 0.46, 0.44, 0.40, 0.38, 0.37]
nn_9_4 = [1, 0.81, 0.69, 0.64, 0.56, 0.49, 0.46, 0.43, 0.39, 0.37, 0.36, 0.33]
on_9_4 = [1, 0.48, 0.38, 0.33] # 15 min timesteps
#
# timestep = [15, 30, 45, 60]
# on_new_4 = [0.84, 0.72, 0.63, 0.55]
# nn_new_4 = [0.84, 0.62, 0.47, 0.38]
# p_new_4 =  [0.56, 0.34, 0.24, 0.19]
#
# #new fix using correct 5 min timesteps
# timestep = [5,    10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60]
# on_new_4 = [,,,]
# nn_new_4 = [0.84, 0.75, 0.68, 0.53, 0.48, 0.50, 0.42, 0.39, 0.37, 0.36, 0.34, 0.32] #model624800
# p_new_4 =  [                                                      0.39, 0.36, 0.33]
#
# nn_new_3 = [] #model1755653
#
# timestep = [30, 60, 90, 120]
# nn_new_5 = [0.17, 0.11, ] #model1734435
# nn_new_5 = [0.24, 0.14, , ] #model1734435 doubled
# p_new_5 = []
#
# plt.plot(timestep, on_new_4, color='blue', linestyle='-', label='Operational nowcast')
# plt.plot(timestep, nn_new_4, color='red', linestyle='-', label='ML prediction')
# plt.plot(timestep, p_new_4, color='green', linestyle='-', label='Persistence')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10)
# plt.show()
# plt.close()
#
# pdb.set_trace()
#
# #operational nowcast
# on_fss_1 = [0.84, 0.69, 0.6, 0.54]
# on_fss_2 = [0.77, 0.58, 0.47, 0.4]
# on_fss_4 = [0.67, 0.43, 0.31, 0.24]
# on_fss_8 = [0.52, 0.24, 0.14, 0.1]
# on_fss_mean = np.mean([on_fss_1,on_fss_2,on_fss_4,on_fss_8], axis=0)
# #Persistence
# p_fss_1 = [0.78,0.65,0.57,0.51]
# p_fss_2 = [0.71,0.55,0.46,0.39]
# p_fss_4 = [0.6,0.41,0.31,0.25]
# p_fss_8 = [0.43,0.24,0.15,0.11]
# p_fss_mean = np.mean([p_fss_1,p_fss_2,p_fss_4,p_fss_8], axis=0)
# #model 625308
# fp_fss_1 = [0.77, 0.6, 0.49, 0.41]
# fp_fss_2 = [0.69, 0.48, 0.35, 0.28]
# fp_fss_4 = [0.57, 0.29, 0.17, 0.12]
# fp_fss_8 = [0.38, 0.11, 0.04, 0.02]
# fp_fss_mean = np.mean([fp_fss_1,fp_fss_2,fp_fss_4,fp_fss_8], axis=0)
#
# fp_train_1 = [0.79, 0.65, 0.55, 0.4]
# fp_train_2 = [0.7, 0.52, 0.42, 0.28]
# fp_train_4 = [0.56, 0.36, 0.27, 0.14]
# fp_train_8 = [0.38, 0.2, 0.12, 0.03]
# fp_train_mean = np.mean([fp_train_1, fp_train_2, fp_train_4, fp_train_8], axis=0)
#
# #model 624800
# lp_fss_1 = [0.76, 0.58, 0.47, 0.4]
# lp_fss_2 = [0.67, 0.46, 0.35, 0.28]
# lp_fss_4 = [0.53, 0.29, 0.19, 0.14]
# lp_fss_8 = [0.32, 0.1, 0.05, 0.03]
# lp_fss_mean = np.mean([lp_fss_1,lp_fss_2,lp_fss_4,lp_fss_8], axis=0)
#
# lp_train_1 = [0.77, 0.62, 0.52, 0.45]
# lp_train_2 = [0.68, 0.49, 0.39, 0.32]
# lp_train_4 = [0.53, 0.32, 0.23, 0.17]
# lp_train_8 = [0.35, 0.16, 0.1, 0.06]
# lp_train_mean = np.mean([lp_train_1, lp_train_2,lp_train_4,lp_train_8], axis=0)
#
# #model 665443
# lp_customloss_1 = [0.75, 0.57, 0.47, 0.4]
# lp_customloss_2 = [0.66, 0.45, 0.34, 0.28]
# lp_customloss_4 = [0.52, 0.28, 0.19, 0.14]
# lp_customloss_8 = [0.32, 0.1, 0.05, 0.03]
# lp_customloss_mean = np.mean([lp_customloss_1,lp_customloss_2,lp_customloss_4,lp_customloss_8], axis=0)
#
# #model 712068
# unet_fss_1 = [0.58, 0.28, 0.18, 0.15]
# unet_fss_2 = [0.54, 0.21, 0.11, 0.08]
# unet_fss_4 = [0.47, 0.14, 0.05, 0.03]
# unet_fss_8 = [0.43, 0.1, 0.02, 0.01]
# unet_fss_mean = np.mean([unet_fss_1, unet_fss_2, unet_fss_4, unet_fss_8], axis=0)
#
# # Plot timeseries
# plt.plot(timestep, p_fss_1, color='green', linestyle='-', label='Per 1mm/hr')
# plt.plot(timestep, p_fss_2, color='green', linestyle='--', label='Per 2mm/hr')
# plt.plot(timestep, p_fss_4, color='green', linestyle='-.', label='Per 4mm/hr')
# plt.plot(timestep, p_fss_8, color='green', linestyle=':', label='Per 8mm/hr')
# plt.plot(timestep, lp_fss_1, color='blue', linestyle='-', label='SVG 1mm/hr')
# plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='SVG 2mm/hr')
# plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='SVG 4mm/hr')
# plt.plot(timestep, lp_fss_8, color='blue', linestyle=':', label='SVG 8mm/hr')
# plt.plot(timestep, on_fss_1, color='orange', linestyle='-', label='ON 1mm/hr')
# plt.plot(timestep, on_fss_2, color='orange', linestyle='--', label='ON 2mm/hr')
# plt.plot(timestep, on_fss_4, color='orange', linestyle='-.', label='ON 4mm/hr')
# plt.plot(timestep, on_fss_8, color='orange', linestyle=':', label='ON 8mm/hr')
# plt.plot(timestep, unet_fss_1, color='red', linestyle='-', label='Unet 1mm/hr')
# plt.plot(timestep, unet_fss_2, color='red', linestyle='--', label='Unet 2mm/hr')
# plt.plot(timestep, unet_fss_4, color='red', linestyle='-.', label='Unet 4mm/hr')
# plt.plot(timestep, unet_fss_8, color='red', linestyle=':', label='Unet 8mm/hr')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10, ncol=4)
# plt.show()
# plt.close()
#
# plt.plot(timestep, lp_train_mean, color='blue', linestyle='--', label='train')
# plt.plot(timestep, lp_fss_mean, color='red', linestyle='--', label='validation')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10)
# plt.show()
# plt.close()
#
# plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='ML model (1mm/hr)')
# plt.plot(timestep, on_fss_1, color='orange', linestyle=':', label='Operational nowcast (1mm/hr)')
# plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='ML model (8mm/hr)')
# plt.plot(timestep, on_fss_8, color='orange', linestyle='-', label='Operational nowcast (8mm/hr)')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10, ncol=2)
# plt.show()
# plt.close()
#
# plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='LP 1mm/hr')
# plt.plot(timestep, fp_fss_1, color='lime', linestyle=':', label='FP 1mm/hr')
# plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='LP 2mm/hr')
# plt.plot(timestep, fp_fss_2, color='lime', linestyle='--', label='FP 2mm/hr')
# plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='LP 4mm/hr')
# plt.plot(timestep, fp_fss_4, color='lime', linestyle='-.', label='FP 4mm/hr')
# plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='LP 8mm/hr')
# plt.plot(timestep, fp_fss_8, color='lime', linestyle='-', label='FP 8mm/hr')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10, ncol=4)
# plt.show()
# plt.close()
#
# plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='MSE 1mm/hr')
# plt.plot(timestep, lp_customloss_1, color='fuchsia', linestyle=':', label='Custom loss 1mm/hr')
# plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='MSE 2mm/hr')
# plt.plot(timestep, lp_customloss_2, color='fuchsia', linestyle='--', label='Custom loss 2mm/hr')
# plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='MSE 4mm/hr')
# plt.plot(timestep, lp_customloss_4, color='fuchsia', linestyle='-.', label='Custom loss 4mm/hr')
# plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='MSE 8mm/hr')
# plt.plot(timestep, lp_customloss_8, color='fuchsia', linestyle='-', label='Custom loss 8mm/hr')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10, ncol=4, loc='upper right')
# plt.show()
# plt.close()
#
#
# plt.plot(timestep, lp_fss_mean, color='blue', linestyle='-', label='SVG model')
# plt.plot(timestep, on_fss_mean, color='orange', linestyle='-', label='Operational nowcast')
# plt.plot(timestep, p_fss_mean, color='green', linestyle='-', label='Persistence')
# plt.plot(timestep, unet_fss_mean, color='red', linestyle='-', label='U-Net')
# plt.xlabel('Time (minutes)')
# plt.ylabel('FSS')
# plt.legend(fontsize=10, ncol=4)
# plt.show()
# plt.close()
