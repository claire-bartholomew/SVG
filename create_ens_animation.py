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
    dt_str = '201909291300' #201908141630' #201909291300'
    prior = 'lp'
    model_path = '/scratch/cbarth/phd/'
    model1 = 'model131219.pth'
    model2 = 'model585435.pth'
    model3 = 'model566185.pth'
    model4 = 'model582525.pth'
    model5 = 'model590512.pth'
    #--------------------------------------------------------------

    if prior == 'fp':
        import run_svg_fp as run_svg
    else:
        import run_svg_lp as run_svg

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    # if files already exist, can comment out this line. If need to run it, need to run from bash terminal.
    #for model in [model1, model2, model3, model4, model5]:
    #    run_svg.main(dt, model_path, model)

    # Neural network output
    nn_cubelist1 = load_nn_pred(dt_str, model1)
    nn_cubelist2 = load_nn_pred(dt_str, model2)
    nn_cubelist3 = load_nn_pred(dt_str, model3)
    nn_cubelist4 = load_nn_pred(dt_str, model4)
    nn_cubelist5 = load_nn_pred(dt_str, model5)

    # Radar sequence
    r_cubelist = load_radar(dt, dt_str, sample_points)
    ## Operational nowcast output
    #n_cubelist = load_nowcast(dt_str, sample_points)

    animate(r_cubelist, nn_cubelist1, nn_cubelist2, nn_cubelist3, nn_cubelist4, nn_cubelist5, dt_str, prior)
    #pdb.set_trace()

def animate(r_cubelist, nn_cubelist1, nn_cubelist2, nn_cubelist3, nn_cubelist4, nn_cubelist5, dt_str, prior):
    # define colours and levels for colorbar
    colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
    levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]

    # Set up video configuration
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(artist='Matplotlib')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    # Configure plots
    fig = plt.figure(figsize=(12, 10))

    # Set video filename
    filename = "animations/{}_ens_radar_animation_{}.mp4".format(prior, dt_str)

    # Create plot frames
    with writer.saving(fig, filename, 300):
        for t in range(23): #8):

            for a, arr in enumerate([nn_cubelist2[0][t+1], nn_cubelist3[0][t+1], nn_cubelist4[0][t+1], nn_cubelist5[0][t+1]]):
                if a == 0:
                    max_arr = np.maximum(nn_cubelist1[0][t+1].data, arr.data)
                    #mask = np.where()
                else:
                    max_arr = np.maximum(max_arr, arr.data)
            max_cube = nn_cubelist1[0][t+1].copy()
            max_cube.data = max_arr

            mean_arr = (nn_cubelist1[0][t+1].data + nn_cubelist1[0][t+1].data + nn_cubelist2[0][t+1].data + 
                        nn_cubelist3[0][t+1].data + nn_cubelist4[0][t+1].data + nn_cubelist5[0][t+1].data) / 5
            mean_cube = nn_cubelist1[0][t+1].copy()
            mean_cube.data = mean_arr

            threshold = 2
            #masked_arr = np.copy(nn_cubelist1[0][t+1])
            masked_arr = np.zeros((np.shape(nn_cubelist1[0][t+1])[0], np.shape(nn_cubelist1[0][t+1])[1]))

            for a, arr in enumerate([nn_cubelist1[0][t+1].data, nn_cubelist2[0][t+1].data, nn_cubelist3[0][t+1].data, nn_cubelist4[0][t+1].data, nn_cubelist5[0][t+1].data]):
                mask = np.where(arr >= threshold)
                masked_arr[mask] += 1
            masked_arr = masked_arr/5.
            prob_cube = nn_cubelist1[0][t+1].copy()
            prob_cube.data = masked_arr

            print('time = ', (t+1)*5) #15)
            ax = fig.add_subplot(3,3,1)
            cf = subplot(r_cubelist[t], 'Radar', levels, colors)
            ax = fig.add_subplot(3,3,2)
            cf = subplot(nn_cubelist1[0][t+1], 'Model 1', levels, colors)
            ax = fig.add_subplot(3,3,3)
            cf = subplot(nn_cubelist2[0][t+1], 'Model 2', levels, colors)
            ax = fig.add_subplot(3,3,4)
            cf = subplot(nn_cubelist3[0][t+1], 'Model 3', levels, colors)
            ax = fig.add_subplot(3,3,5)
            cf = subplot(nn_cubelist4[0][t+1], 'Model 4', levels, colors)
            ax = fig.add_subplot(3,3,6)
            cf = subplot(nn_cubelist5[0][t+1], 'Model 5', levels, colors)

            ax = fig.add_subplot(3,3,7)
            cf = subplot(mean_cube, 'Mean', levels, colors)
            ax = fig.add_subplot(3,3,8)
            cf = subplot(max_cube, 'Max', levels, colors)
            ax = fig.add_subplot(3,3,9)
            colors2 = ['black', 'blue', 'lime', 'yellow', 'orange', 'red'] #s, 'fuchsia'] #, 'white']
            levels2 = [0, 0.2, 0.4, 0.6, 0.8, 1.] 
            cf2 = subplot(prob_cube, 'Prob > {} mm/hr'.format(threshold), levels2, colors2)

            # Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
            colorbar_axes = fig.add_axes([0.15, 0.05, 0.73, 0.03])
            # Add the colour bar
            cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            cbar.ax.set_xlabel('Rain rate (mm/hr)')

            fig.suptitle('T+{:02d} min'.format((t+1)*5), fontsize=30)  #(t+1)*15
            # Save frame
            writer.grab_frame()
    plt.close()

def subplot(data, title, levels, colors):
    cf = iplt.contourf(data, levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
    cf.cmap.set_over('white')
    plt.gca().coastlines('50m', color='white')
    plt.title(title, fontsize=18)
    return cf

def load_nn_pred(dt_str, model):
    nn_cubelist = []
    nn_f = '/data/cr1/cbarth/phd/SVG/plots_nn_T{}_{}.nc'.format(dt_str, model[:-4]) #model_output/model131219/nn_T{}.nc'.format(dt_str)
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
