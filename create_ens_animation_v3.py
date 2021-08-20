from datetime import datetime, timedelta
import iris
import nimrod_to_cubes as n2c
import numpy as np
import pdb
import os
import matplotlib.animation as manimation
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib
import matplotlib.pyplot as plt


def main():
    #--------------------------------------------------------------
    # Options:
    dt_str = '201908121200' #1100' #201908141630' #201909291300' #201908141630' #201909291300'
    prior = 'lp'
    model_path = '/scratch/cbarth/phd/'
    model = 'model624800.pth'

    #--------------------------------------------------------------

    if prior == 'fp':
        import run_svg_fp as run_svg
    else:
        import run_svg_lp as run_svg

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    domain = [160, 288, 130, 258] # england (training data domain)
    threshold = 64.

    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    # if files already exist, can comment out this line. If need to run it, need to run from bash terminal.
    #for model in [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]:
    #    run_svg.main(dt, model_path, model, domain, threshold)

    ens_cubes = []

    # Neural network output
    for ens_n in range(30):
        nn_cubelist = load_nn_pred(dt_str, model, ens_n)
        ens_cubes.append(nn_cubelist)

    # Radar sequence
    r_cubelist = load_radar(dt, dt_str, sample_points)
    ## Operational nowcast output
    #n_cubelist = load_nowcast(dt_str, sample_points)

    animate(r_cubelist, ens_cubes, dt_str, prior)
    prob_animate(r_cubelist, ens_cubes, dt_str, prior)
    #pdb.set_trace()

def animate(r_cubelist, ens_cubes, dt_str, prior):
    # define colours and levels for colorbar
    colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
    levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]

    # Set up video configuration
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(artist='Matplotlib')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    # Configure plots
    fig = plt.figure(figsize=(17, 8))

    # Set video filename
    filename = "animations/{}_ens2_radar_animation_{}.mp4".format(prior, dt_str)

    # Create plot frames
    with writer.saving(fig, filename, 300):
        for t in range(14):
            print('time = ', (t+1)*5)
            for n, ens in enumerate(ens_cubes):
                ax = fig.add_subplot(4, 8, n+1)
                cf = subplot(ens[0][t+1][0:74, 54:128], '', levels, colors)
            ax = fig.add_subplot(4, 8, 32)
            cf = subplot(r_cubelist[t][0:74, 54:128], 'Radar', levels, colors)

            # Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
            colorbar_axes = fig.add_axes([0.33, 0.08, 0.33, 0.02])
            # Add the colour bar
            cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            cbar.ax.set_xlabel('Rain rate (mm/hr)', fontsize=10)
            #fig.tight_layout()
            fig.suptitle('T+{:02d} min'.format((t+1)*5), fontsize=16)
            # Save frame
            writer.grab_frame()
    plt.tight_layout()
    plt.close()

def prob_animate(r_cubelist, ens_cubes, dt_str, prior):
    # define colours and levels for colorbar
    colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
    levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]

    # Set up video configuration
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(artist='Matplotlib')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    # Configure plots
    fig = plt.figure(figsize=(12, 6))

    # Set video filename
    filename = "animations/{}_ens4_radar_animation_{}.mp4".format(prior, dt_str)

    threshold = 4
    # Create plot frames
    with writer.saving(fig, filename, 300):
        for t in range(14): #8):
            #pdb.set_trace()
            masked_arr = np.zeros((np.shape(ens_cubes[0][0][t+1])[0], np.shape(ens_cubes[0][0][t+1])[1]))
            for a, arr in enumerate(ens_cubes):
                arr2 = arr[0][t+1].data
                if a == 0:
                    max_arr = arr2
                    mean_arr = arr2
                else:
                    max_arr = np.maximum(max_arr, arr2)
                    mean_arr += arr2

                mask = np.where(arr2 >= threshold)
                masked_arr[mask] += 1

            max_cube = ens_cubes[0][0][t+1].copy()
            max_cube.data = max_arr

            mean_arr = mean_arr / len(ens_cubes)
            mean_cube = ens_cubes[0][0][t+1].copy()
            mean_cube.data = mean_arr

            masked_arr = masked_arr/len(ens_cubes)
            prob_cube = ens_cubes[0][0][t+1].copy()
            prob_cube.data = masked_arr

            ax = fig.add_subplot(1, 2, 1)
            #Mask radar data below threshold
            r_data = r_cubelist[t].data
            r_data[np.where(r_data<threshold)] = 0
            r_data[np.where(r_data>=threshold)] = 1
            r_cubelist[t].data = r_data
            # Plot radar data
            cf = iplt.contourf(r_cubelist[t][0:64, 64:128] , levels=[0, 0.5, 1], colors=['black', 'yellow']) #, origin='lower', extend='max')
            #cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('Radar rain rate > {} mm/hr'.format(threshold), fontsize=14)
            cbar = plt.colorbar(cf, orientation='horizontal')
            cbar.ax.set_xlabel('Rain > {} mm/hr'.format(threshold), fontsize=14) #rate (mm/hr)')

            ##Plot probability data
            levels2 = [0, 0.2, 0.4, 0.6, 0.8, 1.]
            ax = fig.add_subplot(1, 2, 2)
            cf = iplt.contourf(prob_cube[0:64, 64:128], levels=levels2) #, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('NN probability > {} mm/hr'.format(threshold), fontsize=14)
            cbar = plt.colorbar(cf, orientation='horizontal')
            cbar.ax.set_xlabel('%')

            ##Plot mean/max data
            #ax = fig.add_subplot(1, 3, 2)
            #cf = qplt.contourf(mean_cube, levels, colors=colors, origin='lower', extend='max')
            #cf.cmap.set_over('white')
            #plt.gca().coastlines('50m', color='white')
            #plt.title('Mean', fontsize=18)

            #ax = fig.add_subplot(1, 3, 3)
            #cf = qplt.contourf(max_cube, levels, colors=colors, origin='lower', extend='max')
            #cf.cmap.set_over('white')
            #plt.gca().coastlines('50m', color='white')
            #plt.title('Max', fontsize=18)

            ## Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
            #colorbar_axes = fig.add_axes([0.15, 0.05, 0.73, 0.03])
            ## Add the colour bar
            #cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            #cbar.ax.set_xlabel('Rain rate (mm/hr)')

            fig.suptitle('T+{:02d} min'.format((t+1)*5), fontsize=14)
            # Save frame
            writer.grab_frame()
    plt.close()

def subplot(data, title, levels, colors):
    cf = iplt.contourf(data, levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
    cf.cmap.set_over('white')
    plt.gca().coastlines('50m', color='white')
    plt.title(title, fontsize=12)
    return cf

def load_nn_pred(dt_str, model, ens_n):
    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    dt = dt + timedelta(minutes = 5)
    dt_str = datetime.strftime(dt, '%Y%m%d%H%M')
    nn_cubelist = []
    nn_f = '/data/cr1/cbarth/phd/SVG/model_output/{}_ens3/plots_nn_T{}_{}_ens{}.nc'.format(model[:-4], dt_str, model[:-4], ens_n)
    print(nn_f)
    # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
    cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
    nn_cubes = list(cube_gen)
    nn_cube = nn_cubes[0]
    for t in range(1, 25): #[3, 6, 9, 12, 15, 18, 21, 24]:
        nn_cubelist.append(nn_cube)

    return nn_cubelist

def load_nowcast(dt_str, sample_points):
    # Load nowcast data
    nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
    cubelist = n2c.nimrod_to_cubes(nwcst_f)
    n_cubelist = []
    nowcast = cubelist[2]
    if nowcast.name() != 'rainrate':
        print('rainrate not at index 2')
        for i in len(cubelist):
            if cubelist[i].name() == 'rainrate':
                nowcast = cubelist[i]

    nc_cube = nowcast.interpolate(sample_points, iris.analysis.Linear())
    nowcast_cube = nc_cube[:, 160:288, 130:258]/32
    for cu in [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]: #range(8): #[1, 3]: #, 5]: #i.e. t+30, t+60, t+90
        n_cubelist.append(nowcast_cube[cu])

    return n_cubelist #nowcast_cube #n_cubelist

def load_radar(dt, dt_str, sample_points):
    # Load radar data
    r_cubelist = []
    for t in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]: #[15, 30, 45, 60, 75, 90, 105, 120]
        ti = (dt + timedelta(minutes = t)).strftime('%Y%m%d%H%M')
        radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(ti)
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[160:288, 130:258]/32
        r_cubelist.append(radar_cube)

    return r_cubelist

if __name__ == "__main__":
    main()
