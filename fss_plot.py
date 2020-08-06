import iris
import matplotlib.pyplot as plt
import numpy as np

timestep = [15, 30, 45, 60]

#operational nowcast
on_fss_1 = [0.84, 0.69, 0.6, 0.54]
on_fss_2 = [0.77, 0.58, 0.47, 0.4]
on_fss_4 = [0.67, 0.43, 0.31, 0.24]
on_fss_8 = [0.52, 0.25, 0.14, 0.1]
on_fss_mean = np.mean([on_fss_1,on_fss_2,on_fss_4,on_fss_8], axis=0)
#Persistence
p_fss_1 = []
p_fss_2 = []
p_fss_4 = []
p_fss_8 = []
#p_fss_mean = np.mean([p_fss_1,p_fss_2,p_fss_4,p_fss_8], axis=0)
#model 625308
fp_fss_1 = [0.77, 0.6, 0.49, 0.41]
fp_fss_2 = [0.69, 0.48, 0.35, 0.28]
fp_fss_4 = [0.57, 0.29, 0.17, 0.12]
fp_fss_8 = [0.38, 0.11, 0.04, 0.02]
fp_fss_mean = np.mean([fp_fss_1,fp_fss_2,fp_fss_4,fp_fss_8], axis=0)

fp_train_1 = [0.79, 0.65, 0.55, 0.4]
fp_train_2 = [0.7, 0.52, 0.42, 0.28]
fp_train_4 = [0.56, 0.36, 0.27, 0.14]
fp_train_8 = [0.38, 0.2, 0.12, 0.03]
fp_train_mean = np.mean([fp_train_1, fp_train_2, fp_train_4, fp_train_8], axis=0)

#model 624800
lp_fss_1 = [0.76, 0.58, 0.47, 0.4]
lp_fss_2 = [0.67, 0.46, 0.35, 0.28]
lp_fss_4 = [0.53, 0.29, 0.19, 0.14]
lp_fss_8 = [0.32, 0.1, 0.05, 0.03]
lp_fss_mean = np.mean([lp_fss_1,lp_fss_2,lp_fss_4,lp_fss_8], axis=0)

lp_train_1 = [0.77, 0.62, 0.52, 0.45]
lp_train_2 = [0.68, 0.49, 0.39, 0.32]
lp_train_4 = [0.53, 0.32, 0.23, 0.17]
lp_train_8 = [0.35, 0.16, 0.1, 0.06]
lp_train_mean = np.mean([lp_train_1, lp_train_2,lp_train_4,lp_train_8], axis=0)

#model 665443
lp_customloss_1 = [0.75, 0.57, 0.47, 0.4]
lp_customloss_2 = [0.66, 0.45, 0.34, 0.28]
lp_customloss_4 = [0.52, 0.28, 0.19, 0.14]
lp_customloss_8 = [0.32, 0.1, 0.05, 0.03]
lp_customloss_mean = np.mean([lp_customloss_1,lp_customloss_2,lp_customloss_4,lp_customloss_8], axis=0)

# Plot timeseries
plt.plot(timestep, lp_train_mean, color='blue', linestyle='--', label='train')
plt.plot(timestep, lp_fss_mean, color='red', linestyle='--', label='validation')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10)
plt.show()
plt.close()

plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='LP 1mm/hr')
plt.plot(timestep, fp_fss_1, color='lime', linestyle=':', label='FP 1mm/hr')
plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='LP 2mm/hr')
plt.plot(timestep, fp_fss_2, color='lime', linestyle='--', label='FP 2mm/hr')
plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='LP 4mm/hr')
plt.plot(timestep, fp_fss_4, color='lime', linestyle='-.', label='FP 4mm/hr')
plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='LP 8mm/hr')
plt.plot(timestep, fp_fss_8, color='lime', linestyle='-', label='FP 8mm/hr')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10, ncol=4)
plt.show()
plt.close()
