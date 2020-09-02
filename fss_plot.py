import iris
import matplotlib.pyplot as plt
import numpy as np

timestep = [15, 30, 45, 60]

#operational nowcast
on_fss_1 = [0.84, 0.69, 0.6, 0.54]
on_fss_2 = [0.77, 0.58, 0.47, 0.4]
on_fss_4 = [0.67, 0.43, 0.31, 0.24]
on_fss_8 = [0.52, 0.24, 0.14, 0.1]
on_fss_mean = np.mean([on_fss_1,on_fss_2,on_fss_4,on_fss_8], axis=0)
#Persistence
p_fss_1 = [0.78,0.65,0.57,0.51]
p_fss_2 = [0.71,0.55,0.46,0.39]
p_fss_4 = [0.6,0.41,0.31,0.25]
p_fss_8 = [0.43,0.24,0.15,0.11]
p_fss_mean = np.mean([p_fss_1,p_fss_2,p_fss_4,p_fss_8], axis=0)
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

#model 712068
unet_fss_1 = [0.58, 0.28, 0.18, 0.15]
unet_fss_2 = [0.54, 0.21, 0.11, 0.08]
unet_fss_4 = [0.47, 0.14, 0.05, 0.03]
unet_fss_8 = [0.43, 0.1, 0.02, 0.01]
unet_fss_mean = np.mean([unet_fss_1, unet_fss_2, unet_fss_4, unet_fss_8], axis=0)

# Plot timeseries
plt.plot(timestep, p_fss_1, color='green', linestyle='-', label='Per 1mm/hr')
plt.plot(timestep, p_fss_2, color='green', linestyle='--', label='Per 2mm/hr')
plt.plot(timestep, p_fss_4, color='green', linestyle='-.', label='Per 4mm/hr')
plt.plot(timestep, p_fss_8, color='green', linestyle=':', label='Per 8mm/hr')
plt.plot(timestep, lp_fss_1, color='blue', linestyle='-', label='SVG 1mm/hr')
plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='SVG 2mm/hr')
plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='SVG 4mm/hr')
plt.plot(timestep, lp_fss_8, color='blue', linestyle=':', label='SVG 8mm/hr')
plt.plot(timestep, on_fss_1, color='orange', linestyle='-', label='ON 1mm/hr')
plt.plot(timestep, on_fss_2, color='orange', linestyle='--', label='ON 2mm/hr')
plt.plot(timestep, on_fss_4, color='orange', linestyle='-.', label='ON 4mm/hr')
plt.plot(timestep, on_fss_8, color='orange', linestyle=':', label='ON 8mm/hr')
plt.plot(timestep, unet_fss_1, color='red', linestyle='-', label='Unet 1mm/hr')
plt.plot(timestep, unet_fss_2, color='red', linestyle='--', label='Unet 2mm/hr')
plt.plot(timestep, unet_fss_4, color='red', linestyle='-.', label='Unet 4mm/hr')
plt.plot(timestep, unet_fss_8, color='red', linestyle=':', label='Unet 8mm/hr')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10, ncol=4)
plt.show()
plt.close()



plt.plot(timestep, lp_train_mean, color='blue', linestyle='--', label='train')
plt.plot(timestep, lp_fss_mean, color='red', linestyle='--', label='validation')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10)
plt.show()
plt.close()

plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='ML model (1mm/hr)')
plt.plot(timestep, on_fss_1, color='orange', linestyle=':', label='Operational nowcast (1mm/hr)')
plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='ML model (8mm/hr)')
plt.plot(timestep, on_fss_8, color='orange', linestyle='-', label='Operational nowcast (8mm/hr)')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10, ncol=2)
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

plt.plot(timestep, lp_fss_1, color='blue', linestyle=':', label='MSE 1mm/hr')
plt.plot(timestep, lp_customloss_1, color='fuchsia', linestyle=':', label='Custom loss 1mm/hr')
plt.plot(timestep, lp_fss_2, color='blue', linestyle='--', label='MSE 2mm/hr')
plt.plot(timestep, lp_customloss_2, color='fuchsia', linestyle='--', label='Custom loss 2mm/hr')
plt.plot(timestep, lp_fss_4, color='blue', linestyle='-.', label='MSE 4mm/hr')
plt.plot(timestep, lp_customloss_4, color='fuchsia', linestyle='-.', label='Custom loss 4mm/hr')
plt.plot(timestep, lp_fss_8, color='blue', linestyle='-', label='MSE 8mm/hr')
plt.plot(timestep, lp_customloss_8, color='fuchsia', linestyle='-', label='Custom loss 8mm/hr')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10, ncol=4, loc='upper right')
plt.show()
plt.close()


plt.plot(timestep, lp_fss_mean, color='blue', linestyle='-', label='SVG model')
plt.plot(timestep, on_fss_mean, color='orange', linestyle='-', label='Operational nowcast')
plt.plot(timestep, p_fss_mean, color='green', linestyle='-', label='Persistence')
plt.plot(timestep, unet_fss_mean, color='red', linestyle='-', label='U-Net')
plt.xlabel('Time (minutes)')
plt.ylabel('FSS')
plt.legend(fontsize=10, ncol=4)
plt.show()
plt.close()
