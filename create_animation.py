from datetime import datetime, timedelta
import iris
import nimrod_to_cubes as n2c
import numpy as np
import pdb
import run_svg
import os
import matplotlib.animation as manimation
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib
import matplotlib.pyplot as plt

def main():
    #--------------------------------------------------------------
    # Options:
    dt_str = '201909291300'
    model_path = '/scratch/cbarth/phd/model131219.pth'
    #--------------------------------------------------------------

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    run_svg.main(dt, trained_model)

    # Neural network output
    nn_cubelist = load_nn_pred(dt_str)
    # Radar sequence
    r_cubelist = load_radar(dt, dt_str, sample_points)
    # Operational nowcast output
    n_cubelist = load_nowcast(dt_str, sample_points)

    animate(r_cubelist, n_cubelist, nn_cubelist, dt_str)
    #pdb.set_trace()

def animate(r_cubelist, n_cubelist, nn_cubelist, dt_str):
    # define colours and levels for colorbar
    colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
    levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]

    # Set up video configuration
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(artist='Matplotlib')
    writer = FFMpegWriter(fps=2, metadata=metadata)
    # Configure plots
    fig = plt.figure(figsize=(12, 5))

    # Set video filename
    filename = "radar_animation_{}.mp4".format(dt_str)

    # Create plot frames
    with writer.saving(fig, filename, 300):
        for t in range(8):
            print('time = ', (t+1)*15)
            ax = fig.add_subplot(1,3,1)
            cf = iplt.contourf(r_cubelist[t], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('Radar', fontsize=18)
            ax = fig.add_subplot(1,3,2)
            #pdb.set_trace()
            cf = iplt.contourf(nn_cubelist[0][t+1], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('SVG prediction', fontsize=18)
            ax = fig.add_subplot(1,3,3)
            cf = iplt.contourf(n_cubelist[t], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('Op nowcast', fontsize=18)

            # Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
            colorbar_axes = fig.add_axes([0.15, 0.1, 0.73, 0.03])
            # Add the colour bar
            cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            cbar.ax.set_xlabel('Rain rate (mm/hr)')

            fig.suptitle('T+{:02d} min'.format((t+1)*15), fontsize=30)
            # Save frame
            writer.grab_frame()
    plt.close()



def load_nn_pred(dt_str):
    nn_cubelist = []
    nn_f = '/data/cr1/cbarth/phd/SVG/plots_nn_T{}.nc'.format(dt_str) #model_output/model131219/nn_T{}.nc'.format(dt_str)
    print(nn_f)
    # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
    cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
    nn_cubes = list(cube_gen)
    nn_cube = nn_cubes[0]
    for t in [3, 6, 9, 12, 15, 18, 21, 24]:
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
    #for cu in range(8): #[1, 3]: #, 5]: #i.e. t+30, t+60, t+90
    #    n_cubelist.append(nowcast_cube[cu])

    return nowcast_cube #n_cubelist

def load_radar(dt, dt_str, sample_points):
    # Load radar data
    r_cubelist = []
    for t in [15, 30, 45, 60, 75, 90, 105, 120]:
        ti = (dt + timedelta(minutes = t)).strftime('%Y%m%d%H%M')
        radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(ti)
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[160:288, 130:258]/32
        r_cubelist.append(radar_cube)

    return r_cubelist

if __name__ == "__main__":
    main()
