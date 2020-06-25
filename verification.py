from datetime import datetime, timedelta
import iris
import nimrod_to_cubes as n2c
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

def main():

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    #---------------------------- OPTIONS --------------------------------#
    # Load all dates in op_nowcast data
    files = [f'/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/2019{mo:02}{dd:02}{h:02}{mi:02}_u1096_ng_pp_precip_2km' for mi in range(0, 60, 15)\
             for h in range(24) for dd in range(1, 32) for mo in range(8, 11)]

    # Choose variables
    thrshld = 1  # rain rate threshold (mm/hr)
    neighbourhood = 9 #25   # neighbourhood size (e.g. 9 = 3x3)
    timesteps = [60] #30, 60]
    #---------------------------------------------------------------------#

    files_exist = []
    for file in files:
        if os.path.isfile(file):
            files_exist.append(file)

    for i, flt in enumerate(timesteps):
        fbs_nn_sum = 0
        fbs_nn_worst_sum = 0
        fbs_on_sum = 0
        fbs_on_worst_sum = 0
        count_files = 0
        all_radar = []
        all_nn = []
        all_on = []
        for file in files_exist:
            #print(file)
            dt = datetime.strptime(file, '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/%Y%m%d%H%M_u1096_ng_pp_precip_2km')
            dt_str = dt.strftime('%Y%m%d%H%M')

            # Load data and calculate FBS scores:
            # Neural network output
            nn_cubelist, skip0 = load_nn_pred(dt_str, timesteps)
            # Operational nowcast output
            n_cubelist, skip = load_nowcast(dt_str, sample_points, timesteps)

            if ((skip0 == False) & (skip == False)):
                count_files += 1
                r_cubelist = load_radar(dt, dt_str, sample_points, timesteps)
                # Generate fractions over grid
                ob_fraction = generate_fractions(r_cubelist[i],
                                        n_size=neighbourhood, threshold=thrshld)
                nn_nc_fraction = generate_fractions(nn_cubelist[i],
                                        n_size=neighbourhood, threshold=thrshld)
                on_nc_fraction = generate_fractions(n_cubelist[i],
                                        n_size=neighbourhood, threshold=thrshld)

                # Calculate FBS and FBSworst
                fbs, fbs_worst = calculate_fbs(ob_fraction, nn_nc_fraction)
                fbs_nn_sum += fbs
                fbs_nn_worst_sum += fbs_worst
                fbs_on, fbs_worst_on = calculate_fbs(ob_fraction, on_nc_fraction)
                fbs_on_sum += fbs_on
                fbs_on_worst_sum += fbs_worst_on

                # Calculate all data values for generating PDFs
                all_radar.append(r_cubelist[i].data)
                all_nn.append(nn_cubelist[i].data)
                all_on.append(n_cubelist[i].data)
       
        # Calculate FSS (following method in Roberts (2008))
        fss_nn = 1 - fbs_nn_sum /fbs_nn_worst_sum
        print('FSS for NN at t+{} = {}'.format(flt, fss_nn))
        fss_on = 1 - fbs_on_sum /fbs_on_worst_sum
        print('FSS for Op Ncst at t+{} = {}'.format(flt, fss_on))

    print('number of files: {}'.format(count_files))

    generate_pdf(all_radar, all_nn, all_on)

    #pdb.set_trace()

def generate_pdf(all_radar, all_nn, all_on):
    '''
    Plot histogram comparing distribution of rain rate values in prediction vs truth
    '''
    bins=[0,1,2,4,8,16,32,64]

    radar_values = np.array(all_radar).flatten()

    nn_values = np.array(all_nn).flatten()

    on_values = np.array(all_on).flatten()

    plt.subplot(1,3,1)
    #plt.hist(radar_values, bins, log=True) #density=True, log=True)
    counts, _, patches = plt.hist(radar_values, bins, log=True) #density=True, log=True)
    for i, xy in enumerate(zip(bins, counts)): plt.annotate('%s' % counts[i], xy=xy, textcoords='data')
    plt.ylim((1000, 250000000))
    plt.xlabel('Rain rate (mm/hr)')
    plt.title('Observed values')
    plt.subplot(1,3,2)
    #plt.hist(nn_values, bins, log=True) #density=True, log=True)
    counts, _, patches = plt.hist(nn_values, bins, log=True) #density=True, log=True)
    for i, xy in enumerate(zip(bins, counts)): plt.annotate('%s' % counts[i], xy=xy, textcoords='data')
    plt.ylim((1000, 250000000))
    plt.xlabel('Rain rate (mm/hr)')
    plt.title('ML Predicted values')
    plt.subplot(1,3,3)
    counts, _, patches = plt.hist(on_values, bins, log=True) #density=True, log=True)
    for i, xy in enumerate(zip(bins, counts)): plt.annotate('%s' % counts[i], xy=xy, textcoords='data')
    plt.ylim((1000, 250000000))
    plt.xlabel('Rain rate (mm/hr)')
    plt.title('ON Predicted values')
    plt.show()

    pdb.set_trace()


def calculate_fbs(ob_fraction, nc_fraction):
    '''
    Calculate Fractions Skill Score (FSS) using method as in Roberts (2008)
    Args:
        ob_fraction (arr):
        nc_fraction (arr):
    Returns:
        fbs (float):
        fbs_worst (float):
    '''
    n = np.shape(ob_fraction)[0] * np.shape(ob_fraction)[1]
    fbs = 1 / n * np.sum((ob_fraction - nc_fraction)**2)
    fbs_worst = 1 / n * (np.sum(ob_fraction**2) + np.sum(nc_fraction**2))

    return fbs, fbs_worst

def generate_fractions(cube, n_size, threshold):
    '''
    Function to calculate for each pixel the fraction of surrounding pixels
    within a neighbourhood that exceed a specified threshold.
    Args:
        cube (iris cube): rain rate data from radar or nowcast
        n_size (int): sqaure area of neighbourhood in number of pixels (e.g. 25 for 5x5)
        threshold (int): rain rate threshold in mm/hr
    Outputs:
        fractions (numpy array): computed fractions over the grid
    '''
    # Create binarised array using selected threshold
    threshold_data = np.zeros((np.shape(cube.data)))
    condition_met = np.where(cube.data >= threshold)
    threshold_data[condition_met] = 1
    # Create array of smaller size to avoid border issues
    border = int(np.sqrt(n_size) - 1)
    fractions = np.zeros(np.shape(cube.data[border:-border, border:-border]))

    # Create sliding window to calculate fraction of neighbourhood exceeding threshold
    window = int(np.sqrt(n_size))
    for x in range(np.shape(fractions)[0]):
        for y in range(np.shape(fractions)[1]):
            filter = threshold_data[x:x+window, y:y+window]
            fract = np.sum(filter)/n_size
            fractions[x, y] = fract

    return fractions

def load_nn_pred(dt_str, timesteps):
    nn_cubelist = []
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model131219/nn_T{}.nc'.format(dt_str)
    print(nn_f)
    if os.path.exists(nn_f):
        skip = False
        # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
        cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
        nn_cubes = list(cube_gen)
        nn_cube = nn_cubes[0]
        # Get index for timesteps
        for timestep in timesteps:
            nn_cubelist.append(nn_cube[int(timestep / 5)])
    else:
        skip = True

    return nn_cubelist, skip

def load_nowcast(dt_str, sample_points, timesteps):
    # Load nowcast data
    nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
    cubelist = n2c.nimrod_to_cubes(nwcst_f)
    n_cubelist = []
    if len(cubelist) < 3:
        skip = True
    else:
        skip = False
        nowcast = cubelist[2]
        if nowcast.name() != 'rainrate':
            print('rainrate not at index 2')
            for i in len(cubelist):
                if cubelist[i].name() == 'rainrate':
                    nowcast = cubelist[i]

        nc_cube = nowcast.interpolate(sample_points, iris.analysis.Linear())
        nowcast_cube = nc_cube[:, 160:288, 130:258] / 32
        # Get index for timesteps
        for cu in timesteps:
            n_cubelist.append(nowcast_cube[int(cu / 30 * 2 - 1)])

    return n_cubelist, skip

def load_radar(dt, dt_str, sample_points, timesteps):
    # Load radar data
    r_cubelist = []
    for t in timesteps:
        ti = (dt + timedelta(minutes = t)).strftime('%Y%m%d%H%M')
        radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[160:288, 130:258] / 32
        r_cubelist.append(radar_cube)

    return r_cubelist

if __name__ == "__main__":
    main()
