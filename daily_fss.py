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

def main(leadtime):

    #define datetime of analysis time (i.e. T+0)
    #dt_str = '201901170200'
    count = 0
    model_n = '624800'
    ts = 5
    thrshld = 1 # rain rate threshold for FSS (mm/hr)
    neighbourhood = 9  # neighbourhood size (e.g. 9 = 3x3)
    #leadtime = 30 # forecast lead time
    domain = [160, 288, 130, 258] # england (training data domain)
    mean_rain = 0 #0.01 #0.003 #threshold for selecting rainy days
    df = make_df()
    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    start_date = datetime(2019, 1, 1, 0)
    end_date = datetime(2020, 1, 1, 0)
    delta = timedelta(hours=1)
    while start_date <= end_date:
        #print(start_date)
        dt_str = start_date.strftime('%Y%m%d%H%M')
        files_exist = check_files(dt_str)
        if files_exist:
            r_cube = load_radar(dt_str, sample_points, leadtime, domain)
            mean_data = np.mean(r_cube.data)
            print('mean rain = ', mean_data)
            if mean_data > mean_rain:
                print(dt_str)
                count, df = add_to_df(count, r_cube, dt_str, sample_points, leadtime, ts,
                             domain, thrshld, model_n, neighbourhood, df, mean_rain, mean_data)
        start_date += delta

    df.to_csv('/data/cr1/cbarth/phd/SVG/daily_fss4_{}_n{}_th{}.csv'.format(leadtime, neighbourhood, thrshld))

    #pdb.set_trace()


def add_to_df(count, r_cube, dt_str, sample_points, leadtime, ts, domain, thrshld, model_n, neighbourhood, df, mean_rain, mean_data):
    count += 1
    on_cube = load_op_nowcast(dt_str, sample_points, leadtime, ts, domain)
    p_cube = load_persistence(dt_str, sample_points, domain)
    #nn_cube = load_nn_pred(dt_str, leadtime, model_n, ts, ens_n)
    #Calculate fractions in radar data
    ob_fraction = generate_fractions(r_cube, n_size=neighbourhood, threshold=thrshld)
    on_fss = calculate_fss(on_cube, ob_fraction, neighbourhood, thrshld)
    p_fss = calculate_fss(p_cube, ob_fraction, neighbourhood, thrshld)

    ens_nn_fraction = 0
    ens_fss = []
    for ens_n in range(30):
        nn_cube = load_nn_pred(dt_str, leadtime, model_n, ts, ens_n)
        nn_fraction = generate_fractions(nn_cube, n_size=neighbourhood, threshold=thrshld)
        ens_nn_fraction += nn_fraction
        fbs, fbs_worst = calculate_fbs(ob_fraction, nn_fraction)
        # Calculate FSS
        fss = 1 - fbs / fbs_worst
        ens_fss.append(fss)
        #quickplot(nn_cube, r_cube)

    ens_mean_fraction = ens_nn_fraction / 30.
    fbs_e, fbs_e_worst = calculate_fbs(ob_fraction, ens_mean_fraction)
    # Calculate FSS for NN ensemble
    fss_e_nn = 1 - fbs_e / fbs_e_worst

    df.loc[count] = [dt_str, model_n, count, thrshld, mean_rain, mean_data, leadtime, neighbourhood,
                    fss_e_nn, on_fss, p_fss, ens_fss[0], ens_fss[1], ens_fss[2],
                    ens_fss[3], ens_fss[4], ens_fss[5], ens_fss[6], ens_fss[7],
                    ens_fss[8], ens_fss[9], ens_fss[10], ens_fss[11], ens_fss[12],
                    ens_fss[13], ens_fss[14], ens_fss[15], ens_fss[16], ens_fss[17],
                    ens_fss[18], ens_fss[19], ens_fss[20], ens_fss[21], ens_fss[22],
                    ens_fss[23], ens_fss[24], ens_fss[25], ens_fss[26], ens_fss[27],
                    ens_fss[28], ens_fss[29]]

    #pdb.set_trace()

    return count, df

def calculate_fss(cube, ob_fraction, neighbourhood, thrshld):
    fraction = generate_fractions(cube, n_size=neighbourhood, threshold=thrshld)
    # Calculate FBS and FBSworst
    fbs, fbs_worst = calculate_fbs(ob_fraction, fraction)
    # Calculate FSS
    fss = 1 - fbs / fbs_worst

    return fss

#===============================================================================
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

def check_files(yyyymmddhhmm):
    #convert to datetime and select datetime for nn predictions (offset by 5 mins)
    date_time = datetime.strptime(yyyymmddhhmm, "%Y%m%d%H%M")
    nn_date_time = date_time + timedelta(minutes=5)
    nn_yyyymmddhhmm = nn_date_time.strftime('%Y%m%d%H%M')
    #file locations
    radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(yyyymmddhhmm)
    op_nowcast_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast_5min/{}_u1096_ng_pp_precip5min_2km'.format(yyyymmddhhmm)
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model624800_ens3/plots_nn_T{}_model624800_ens0.nc'.format(nn_yyyymmddhhmm)
    files = [radar_f, op_nowcast_f, nn_f]
    #check files exist
    existing = [f for f in files if os.path.isfile(f)]
    if len(existing) == 3:
        files_exist = True
    else:
        files_exist = False

    return files_exist

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

def load_nn_pred(dt_str, leadtime, model_n, ts, ens_n):
    #need to add 5 mins to datestr to get nn filename
    date = datetime.strptime(dt_str, '%Y%m%d%H%M')
    nn_file_date = date + timedelta(minutes = 5)
    nn_dt_str = datetime.strftime(nn_file_date, '%Y%m%d%H%M')
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model{}_ens3/plots_nn_T{}_model{}_ens{}.nc'.format(model_n, nn_dt_str, model_n, ens_n)
    # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
    cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
    # Get index for leadtime and extract data
    nn_cube = list(cube_gen)[0][int(leadtime / ts)]

    return nn_cube

def load_op_nowcast(dt_str, sample_points, leadtime, ts, domain):
    # Load nowcast data
    nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast_5min/{}_u1096_ng_pp_precip5min_2km'.format(dt_str)
    cubelist = n2c.nimrod_to_cubes(nwcst_f)
    nc_cube = cubelist[0].interpolate(sample_points, iris.analysis.Linear())
    nowcast_cube = nc_cube[:, domain[0]:domain[1], domain[2]:domain[3]] / 32
    # Get index for leadtimes
    on_cube = nowcast_cube[int(leadtime / ts - 1)]
    # Set units (ideally would want to change forecast_reference_time to be dt_str)
    on_cube.units = 'mm/hr'

    return on_cube

def load_persistence(dt_str, sample_points, domain):
    # Persistence forecast
    persist_radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
    p_radar = iris.load(persist_radar_f)
    p_cube = p_radar[0].interpolate(sample_points, iris.analysis.Linear())
    persist_cube = p_cube[domain[0]:domain[1], domain[2]:domain[3]] / 32
    persist_cube.units = 'mm/hr'

    return persist_cube

def load_radar(dt_str, sample_points, leadtime, domain):
    # Load radar data
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    r_dt_str = (dt + timedelta(minutes = leadtime)).strftime('%Y%m%d%H%M')
    radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(r_dt_str)
    radar = iris.load(radar_f)
    r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
    radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]] / 32
    radar_cube.units = 'mm/hr'

    return radar_cube

def make_df():
    df = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0]], columns=['datetime', 'model',
                        'count', 'threshold', 'mean_thrshold', 'mean_data', 'timestep',
                        'neighbourhood', 'fss_e_mean', 'fss_on', 'fss_p', 'fss_e0',
                        'fss_e1', 'fss_e2', 'fss_e3', 'fss_e4', 'fss_e5', 'fss_e6',
                        'fss_e7', 'fss_e8', 'fss_e9', 'fss_e10', 'fss_e11', 'fss_e12',
                        'fss_e13', 'fss_e14', 'fss_e15', 'fss_e16', 'fss_e17',
                        'fss_e18', 'fss_e19', 'fss_e20', 'fss_e21', 'fss_e22',
                        'fss_e23', 'fss_e24', 'fss_e25', 'fss_e26', 'fss_e27',
                        'fss_e28', 'fss_e29'])
    return df

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
    for leadtime in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]: #[15, 30, 45, 60]:
        main(leadtime)
