from datetime import datetime, timedelta
import iris
import nimrod_to_cubes as n2c
import numpy as np
import pdb

def main():

    mm = 8
    dd = 1
    hh = 1
    min = 30
    dt_str = '2019{:02d}{:02d}{:02d}{:02d}'.format(mm, dd, hh, min)
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')

    #start_date =
    #end_date = 

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]
    # Load data
    n_cubelist = load_nowcast(dt_str, sample_points)
    r_cubelist = load_radar(dt, dt_str, sample_points)

    fbs2 = 0
    fbs_worst2 = 0
    # Generate fractions over grid then calculate FSS
    for i, flt in enumerate([30, 60, 90]):
        ob_fraction = generate_fractions(r_cubelist[i], n_size=9, threshold=1)
        nc_fraction = generate_fractions(n_cubelist[i], n_size=9, threshold=1)
        fbs, fbs_worst, fss = calculate_fss(ob_fraction, nc_fraction)
        print('FSS at t+{} = {}'.format(flt, fss))
        fbs2 += fbs
        fbs_worst2 += fbs_worst

    fss2 = 1 - fbs2 /fbs_worst2

    print('overall FSS = {}'.format(fss2))

    pdb.set_trace()


def calculate_fss(ob_fraction, nc_fraction):
    '''
    Calculate Fractions Skill Score (FSS) using method as in Roberts (2008)
    Args:
        ob_fraction (arr):
        nc_fraction (arr):
    Returns:
        fss (float):
    '''
    n = np.shape(ob_fraction)[0] * np.shape(ob_fraction)[1]
    fbs = 1 / n * np.sum((ob_fraction - nc_fraction)**2)
    fbs_worst = 1 / n * (np.sum(ob_fraction**2) + np.sum(nc_fraction**2))
    fss = 1 - fbs / fbs_worst

    return fbs, fbs_worst, fss

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

def load_nowcast(dt_str, sample_points):
    # Load nowcast data
    nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
    cubelist = n2c.nimrod_to_cubes(nwcst_f)
    nowcast = cubelist[2]
    if nowcast.name() != 'rainrate':
        print('rainrate not at index 2')
        for i in len(cubelist):
            if cubelist[i].name() == 'rainrate':
                nowcast = cubelist[i]

    nc_cube = nowcast.interpolate(sample_points, iris.analysis.Linear())
    nowcast_cube = nc_cube[:, 160:288, 130:258]/32
    n_cubelist = []
    for cu in [1, 3, 5]: #i.e. t+30, t+60, t+90
        n_cubelist.append(nowcast_cube[cu])

    return n_cubelist

def load_radar(dt, dt_str, sample_points):
    # Load radar data
    r_cubelist = []
    for t in [30, 60, 90]:
        ti = (dt + timedelta(minutes = t)).strftime('%Y%m%d%H%M')
        radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[160:288, 130:258]/32
        r_cubelist.append(radar_cube)

    return r_cubelist

if __name__ == "__main__":
    main()
