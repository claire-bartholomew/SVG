import numpy as np
import pdb
from datetime import datetime, timedelta
import os
import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import nimrod_to_cubes as n2c


def main():

    #model_n = '624800' #842306' #625308' #624800' #'131219'
    # Choose variables
    thrshld = 1 # rain rate threshold (mm/hr)
    timestep = 45
    domain = [160, 288, 130, 258] # england (training data domain)

    radar_dir = '/data/cr1/cbarth/phd/SVG/verification_data/radar/'
    files = [f'{radar_dir}2019{mo:02}{dd:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' for mo in [1, 5, 9]\
             for dd in range(1, 30) for h in range(24) for mi in range(0, 60, 15)] #[0]]

    probability_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pod_list = []
    far_list = []

    for prob_thsld in probability_thresholds:
        pod, far = roc_plot_data(files, radar_dir, timestep, domain, thrshld, prob_thsld)
        pod_list.append(pod)
        far_list.append(far)

    plot_roc(pod_list, far_list)
    pdb.set_trace()


def roc_plot_data(files, radar_dir, timestep, domain, thrshld, prob_thsld):
    c_table = [0, 0, 0, 0] #hits, misses, false alarms, correct rejections
    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]
    count = 0
    for file in files:
        dt = datetime.strptime(file, '{}%Y%m%d%H%M_nimrod_ng_radar_rainrate_composite_1km_UK'.format(radar_dir))
        dt_str = dt.strftime('%Y%m%d%H%M')
        # Load radar data
        r_cube = load_radar(dt, sample_points, timestep, domain)
        ens_data = np.zeros((np.shape(r_cube.data)))
        # Check if enough rain to be worth verifying
        if np.mean(r_cube.data) > 0.1:
            print(dt)
            for model_n in ['131219', '530043', '566185', '582525', '585435',
                            '590512', '601712', '848512', '624800', '876319']:
                cube, skip = load_nn_pred(dt_str, timestep, model_n)
                if skip == False:
                    # Mask predictions using threshold
                    condition_met = np.where(cube.data >= thrshld)
                    ens_data[condition_met] += 1
                    skip_date = False
                else:
                    skip_date = True
            if skip_date == False:
                # Generate ensemble probability
                ens_data /= 10.

                # Mask radar data
                r_data = np.zeros((np.shape(r_cube.data)))
                condition_met = np.where(r_cube.data >= thrshld)
                r_data[condition_met] += 1

                # Generate contigency table for probability threshold of choice
                hit, miss, fa, cr = contigency_table(ens_data, r_data, prob_thsld)
                c_table[0] += hit
                c_table[1] += miss
                c_table[2] += fa
                c_table[3] += cr

                count += 1

    pod = c_table[0]/(c_table[0]+c_table[1])
    far = c_table[2]/(c_table[0]+c_table[2])
    print('prob threshold = ', prob_thsld)
    print('POD = ', pod)
    print('FAR = ', far)

    print('count = ', count)

    return pod, far

def contigency_table(ens_data, r_data, prob_thsld):
    threshold_data = np.zeros((np.shape(ens_data)))
    threshold_data += 100
    thsld_exceeded = np.where(ens_data >= prob_thsld)
    threshold_data[thsld_exceeded] += 2

    diff = threshold_data - r_data

    hit = (diff == 101).sum()
    miss = (diff == 99.).sum()
    false_alarm = (diff == 102).sum()
    corr_rej = (diff == 100).sum()

    return hit, miss, false_alarm, corr_rej

def load_nn_pred(dt_str, timestep, model_n):
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model{}/plots_nn_T{}_model{}.nc'.format(model_n, dt_str, model_n)
    if os.path.exists(nn_f):
        skip = False
        # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
        cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
        nn_cubes = list(cube_gen)
        nn_cube1 = nn_cubes[0]
        # Get index for timestep and extract data
        nn_cube = nn_cube1[int(timestep / 5)]
    else:
        skip = True

    return nn_cube, skip

def load_radar(dt, sample_points, timestep, domain):
    # Load radar data
    dt_str = (dt + timedelta(minutes = timestep)).strftime('%Y%m%d%H%M')
    radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
    radar = iris.load(radar_f)
    r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
    radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]] / 32
    radar_cube.units = 'mm/hr'

    return radar_cube

def plot_roc(pod, far):
    # Could expand for different rain rate thresholds
    plt.plot(far, pod)
    plt.plot(far, far, color='gray', linestyle='--')
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Probability of Detection')
    plt.title('ROC plot')
    #plt.legend(fontsize=10)
    plt.show()

if __name__ == "__main__":
    main()
