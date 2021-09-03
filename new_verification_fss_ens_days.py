import numpy as np
import pdb
from datetime import datetime, timedelta
import os
import iris
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt
import nimrod_to_cubes as n2c
import pandas as pd

def main(df, leadtime, thrshld, neighbourhood, mean_rain): #, model_n, thrshld, ts, neighbourhood):

    count = 0

    model_n = '624800' #1755653' #624800' #1734435' #624800' #842306' #625308' #624800' #'131219'
    ts = 5 #5 #30 #model timestep separation
    ## Choose variables
    #thrshld = 1 #0.5  #1 # rain rate threshold (mm/hr)
    #neighbourhood = 25 #9 #9 #36 #81 #36 #25   # neighbourhood size (e.g. 9 = 3x3)
    ###leadtime = 30 # forecast lead time
    domain = [160, 288, 130, 258] # england (training data domain)
    #mean_rain = 0.5 #threshold for selecting rainy days

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    radar_dir = '/data/cr1/cbarth/phd/SVG/verification_data/radar/'
    files = [f'{radar_dir}2019{mo:02}{dd:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' for mo in range(6, 7)\
             for dd in range(2, 29) for h in range(24) for mi in range(0, 60, 15)] #[0]]
    #files = [f'{radar_dir}201905{dd:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK'\
    #         for dd in range(2, 30) for h in range(24) for mi in range(10, 70, 15)] #[0]]

    # fbs_nn_sum = 0
    # fbs_nn_worst_sum = 0
    # fbs_nn_e_sum = 0
    # fbs_nn_e_worst_sum = 0
    # fbs_on_sum = 0
    # fbs_on_worst_sum = 0
    # fbs_p_sum = 0
    # fbs_p_worst_sum = 0

    for file in files:
        fbs_nn_sum = 0
        fbs_nn_worst_sum = 0
        fbs_nn_e_sum = 0
        fbs_nn_e_worst_sum = 0
        fbs_on_sum = 0
        fbs_on_worst_sum = 0
        fbs_p_sum = 0
        fbs_p_worst_sum = 0
        dt = datetime.strptime(file, '{}%Y%m%d%H%M_nimrod_ng_radar_rainrate_composite_1km_UK'.format(radar_dir))
        dt_str = dt.strftime('%Y%m%d%H%M')
        #print(dt)
        # Load radar data
        r_cube = load_radar(dt, sample_points, leadtime, domain, ts)
        #print(r_cube)
        if r_cube != 0:
            # Check if enough rain to be worth verifying
            if np.mean(r_cube.data) > mean_rain:
                print(dt)
                #print(r_cube)
                nn_cube, skip = load_nn_pred_det(dt, leadtime, model_n, ts)
                on_cube, skip0 = load_op_nowcast(dt_str, sample_points, leadtime, domain)
                p_cube = load_persistence(dt, sample_points, ts, domain)
                #pdb.set_trace()
                if ((skip == False) & (skip0 == False)):
                    #print(dt)
                    count += 1
                    #quickplot(nn_cube, r_cube)

                    # Generate fractions over grid
                    ob_fraction = generate_fractions(r_cube, n_size=neighbourhood,
                                                 threshold=thrshld)
                    on_fraction = generate_fractions(on_cube, n_size=neighbourhood,
                                                    threshold=thrshld)
                    p_fraction = generate_fractions(p_cube, n_size=neighbourhood,
                                                threshold=thrshld)
                    nn_fraction = generate_fractions(nn_cube, n_size=neighbourhood,
                                                 threshold=thrshld)
                    ens_nn_fraction = 0
                    for ens_n in range(30):
                        nn_cube, skip = load_nn_pred(dt_str, leadtime, model_n, ts, ens_n)
                        e_nn_fraction = generate_fractions(nn_cube, n_size=neighbourhood,
                                                       threshold=thrshld)
                        ens_nn_fraction += e_nn_fraction

                    ens_nn_fraction = ens_nn_fraction / 30.

                    # Calculate FBS and FBSworst for NN
                    fbs, fbs_worst = calculate_fbs(ob_fraction, nn_fraction)
                    fbs_nn_sum += fbs
                    fbs_nn_worst_sum += fbs_worst
                    # Calculate FBS and FBSworst for ensemble NN
                    fbs_e, fbs_e_worst = calculate_fbs(ob_fraction, ens_nn_fraction)
                    fbs_nn_e_sum += fbs_e
                    fbs_nn_e_worst_sum += fbs_e_worst
                    # Calculate FBS and FBSworst for ON
                    fbs_on, fbs_worst_on = calculate_fbs(ob_fraction, on_fraction)
                    fbs_on_sum += fbs_on
                    fbs_on_worst_sum += fbs_worst_on
                    # Calculate FBS and FBSworst for persistence
                    fbs_p, fbs_worst_p = calculate_fbs(ob_fraction, p_fraction)
                    fbs_p_sum += fbs_p
                    fbs_p_worst_sum += fbs_worst_p

                    # Calculate FSS for NN ensemble
                    fss_e_nn = 1 - fbs_nn_e_sum / fbs_nn_e_worst_sum
                    # Calculate FSS for ON
                    fss_on = 1 - fbs_on_sum / fbs_on_worst_sum
                    # Calculate FSS for persistence
                    fss_p = 1 - fbs_p_sum / fbs_p_worst_sum

                    df2 = pd.DataFrame([[dt, model_n, count, thrshld, mean_rain, leadtime, neighbourhood,
                        fss_nn, fss_e_nn, fss_on, fss_p]], columns=['datetime',
                        'model', 'count', 'threshold', 'mean_thrshold', 'timestep',
                        'neighbourhood', 'fss_nn', 'fss_enn', 'fss_on', 'fss_p'])
                    df3 = df.append(df2)

    # # for outputs
    # print('model = ', model_n)
    # print('count = ', count)
    # print('threshold = ', thrshld)
    # print('timestep = ', ts)
    # print ('nhood = ', neighbourhood)
    # #print('leadtime, count, fss_p, fss_nn, fss_e_nn')
    # #print('====================================')
    # # Calculate FSS for NN (following method in Roberts (2008))
    # fss_nn = 1 - fbs_nn_sum / fbs_nn_worst_sum
    # print('====================================')
    # print('FSS for NN at t+{} = {}'.format(leadtime, fss_nn))
    # # Calculate FSS for NN ensemble
    # fss_e_nn = 1 - fbs_nn_e_sum / fbs_nn_e_worst_sum
    # print('====================================')
    # print('ENSEMBLE FSS')
    # print('FSS for ensemble NN at t+{} = {}'.format(leadtime, fss_e_nn))
    # # Calculate FSS for ON
    # print(fbs_on_sum, fbs_on_worst_sum)
    # fss_on = 1 - fbs_on_sum / fbs_on_worst_sum
    # print('FSS for ON at t+{} = {}'.format(leadtime, fss_on))
    # print('====================================')
    # # Calculate FSS for persistence
    # fss_p = 1 - fbs_p_sum / fbs_p_worst_sum
    # print('FSS for persistence at t+{} = {}'.format(leadtime, fss_p))
    # print('====================================')

    # df2 = pd.DataFrame([[model_n, count, thrshld, mean_rain, leadtime, neighbourhood,
    #                    fss_nn, fss_e_nn, fss_on, fss_p]], columns=['model',
    #                    'count', 'threshold', 'mean_thrshold', 'timestep',
    #                    'neighbourhood', 'fss_nn', 'fss_enn', 'fss_on', 'fss_p'])
    # df3 = df.append(df2)

    return df3

def load_nn_pred(dt_str, leadtime, model_n, ts, ens_n):
    #need to add 5 mins to datestr to get nn filenmae in model624800 dir
    date = datetime.strptime(dt_str, '%Y%m%d%H%M')   #05  #09
    nn_file_date = date + timedelta(minutes = 5)
    nn_dt_str = datetime.strftime(nn_file_date, '%Y%m%d%H%M')
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model{}_ens3/plots_nn_T{}_model{}_ens{}.nc'.format(model_n, nn_dt_str, model_n, ens_n) #ens2
    if os.path.exists(nn_f):
        #print('nn file: {}'.format(nn_f))
        skip = False
        # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
        cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
        nn_cubes = list(cube_gen)
        nn_cube1 = nn_cubes[0] #* 2
        # Get index for leadtime and extract data
        nn_cube = nn_cube1[int(leadtime / ts)]
    else:
        skip = True
        nn_cube = False

    return nn_cube, skip

def load_nn_pred_det(dt, leadtime, model_n, ts):
    det_dt = dt + timedelta(minutes=ts)
    det_dt_str = det_dt.strftime('%Y%m%d%H%M')
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model{}/plots_nn_T{}_model{}.nc'.format(model_n, det_dt_str, model_n)
    if os.path.exists(nn_f):
        #print('det nn file: {}'.format(nn_f))
        skip = False
        # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
        cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
        nn_cubes = list(cube_gen)
        nn_cube1 = nn_cubes[0] #* 2
        # Get index for leadtime and extract data
        nn_cube = nn_cube1[int(leadtime / ts)]# + 2)]
    else:
        skip = True
        #print('no file exists: ', nn_f)
        nn_cube = False

    return nn_cube, skip

def load_persistence(dt, sample_points, ts, domain):
    # Persistence forecast
    persist_dt = dt # - timedelta(minutes=ts)
    persist_dt_str = persist_dt.strftime('%Y%m%d%H%M')
    persist_radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(persist_dt_str)

    p_radar = iris.load(persist_radar_f)
    p_cube = p_radar[0].interpolate(sample_points, iris.analysis.Linear())
    persist_cube = p_cube[domain[0]:domain[1], domain[2]:domain[3]] / 32

    return persist_cube

def load_op_nowcast(dt_str, sample_points, leadtime, domain):
    # Load nowcast data
    nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
    if os.path.exists(nwcst_f):
        #print('op nowcast file: {}'.format(nwcst_f))
        cubelist = n2c.nimrod_to_cubes(nwcst_f)
        n_cubelist = []
        if len(cubelist) < 3:
            skip = True
            on_cube = 0
        else:
            skip = False
            nowcast = cubelist[2]
            if nowcast.name() != 'rainrate':
                print('rainrate not at index 2')
                for i in len(cubelist):
                    if cubelist[i].name() == 'rainrate':
                        nowcast = cubelist[i]

            nc_cube = nowcast.interpolate(sample_points, iris.analysis.Linear())
            nowcast_cube = nc_cube[:, domain[0]:domain[1], domain[2]:domain[3]] / 32
            # Get index for leadtimes
            on_cube = nowcast_cube[int(leadtime / 30 * 2 - 1)]
            on_cube.units = 'mm/hr'
    else:
        skip = True
        on_cube = 0

    return on_cube, skip

def load_radar(dt, sample_points, leadtime, domain, ts):
    # Load radar data
    dt_str = (dt + timedelta(minutes = leadtime)).strftime('%Y%m%d%H%M')
    radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
    if os.path.exists(radar_f):
        #print('radar file: {}'.format(radar_f))
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]] / 32
        radar_cube.units = 'mm/hr'
    else:
        radar_cube = 0

    return radar_cube

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

def quickplot(nn_cube, r_cube):
    # Quickplot to check output
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    qplt.contourf(nn_cube)
    plt.gca().coastlines()
    plt.title('ML prediction')
    plt.subplot(122)
    qplt.contourf(r_cube)
    plt.gca().coastlines()
    plt.title('Radar')
    iplt.show()

if __name__ == "__main__":
    #main()
    #import sys; sys.path.append('/home/h03/jcheung/python/lib')
    #import toolbox as tb
    #leadtimes = range(0, 75, 15)
    #tb.parallelise(main)(leadtimes)

    df = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0]], columns=['datetime', 'model', 'count',
                      'threshold', 'mean_thrshold', 'timestep', 'neighbourhood',
                      'fss_nn', 'fss_enn', 'fss_on', 'fss_p'])
    #for mean_rain in [0.1, 0.5]:
    #    for threshold in [1, 4, 10]:
    #        for neighbourhood in [9, 25]:  # neighbourhood size (e.g. 9 = 3x3)
    #            for leadtime in [15, 30, 45, 60]:
    #                df = main(df, leadtime, threshold, neighbourhood, mean_rain)
    #df.to_csv('fss_df.csv')

    mean_rain = 0.1
    threshold = 4 #1, 4, 10]:
    neighbourhood = 25 #9, 25]:  # neighbourhood size (e.g. 9 = 3x3)
    leadtime = 60 #15, 30, 45, 60]:
    df = main(df, leadtime, threshold, neighbourhood, mean_rain)
    df.to_csv('short_fss_df_{}.csv'.format(leadtime))

    #pdb.set_trace()
