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
    dt_str = '201910011400' #08121600' # #08121600' #09240700' ##08121500' #201807271730' #201907301500' #201807271730' #201907301200' #201807271630' #201909291300' #07191300' #10272345' #07191300' #10272345' #291300' #201909191200' #201908141730' #201907260000' #201909282100' #201910011200' #201909281300' #201908141630'
    prior = 'lp'
    model_path = '/scratch/cbarth/phd/'
    model = 'model624800.pth' #model1756151.pth' #model624800.pth' #131219.pth' #624800.pth' #810068.pth'  #817118.pth'  #810069.pth' #625308.pth' #624800.pth' #723607.pth' #712068.pth' #665443.pth' #624800.pth' #model131219.pth' #667922.pth' #665443.pth' #25308.pth' #598965.pth' #585435.pth' #566185.pth' #model562947.pth' #model_fp.pth' #model_530043_lp.pth' #model_529994_fp.pth' #model_fp.pth' #131219.pth' #need to also change this in line 32 of run_svg.py
    #model2 = 'model624800.pth' #842306.pth' #848512.pth' #842306.pth'
    #domain = [288, 416, 100, 228] #scotland
    domain = [160, 288, 130, 258] # england (training data domain)
    #r_domain = [185, 263, 155, 233] #reduced domain to avoid border effects
    threshold = 64. #100. #32.
    datadi = '/data/cr1/cbarth/phd/SVG/casestudies' #/data/cr1/cbarth/phd/SVG/verification_data/radar'
    #plotdi = '/data/cr1/cbarth/phd/SVG'
    #--------------------------------------------------------------
    import run_svg_lp_log as run_svg_log
    if prior == 'fp':
        import run_svg_fp as run_svg
    else:
        import run_svg_lp as run_svg

    # x and y coordinate points to regrid to for consistency
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    dt = datetime.strptime(dt_str, '%Y%m%d%H%M')
    # if files already exist, can comment out this line. If need to run it, need to run from bash terminal.
    run_svg.main(dt, model_path, model, domain, threshold, datadi)
    #run_svg_log.main(dt, model_path, model, domain, threshold)
    #run_svg.main(dt, model_path, model2, domain, threshold)
    #pdb.set_trace()

    # Neural network output
    nn_cubelist = load_nn_pred(dt_str, model, domain, datadi) #always domain here even when plotting r_domain, just uncomment line in the function
    #nn_cubelist2 = load_nn_pred(dt_str, model2, domain) #always domain here even when plotting r_domain, just uncomment line in the function
    # Radar sequence
    r_cubelist = load_radar(dt, dt_str, sample_points, domain, datadi) #r_domain)
    # Operational nowcast output
    #n_cubelist = load_nowcast(dt_str, sample_points, domain) #r_domain)

    animate(r_cubelist, nn_cubelist, dt_str, prior, model) #, model2, nn_cubelist2)
    #pdb.set_trace()

def animate(r_cubelist, nn_cubelist, dt_str, prior, model): #, model2, nn_cubelist2):
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
    filename = "animations/{}_radar_animation_{}_{}.mp4".format(prior, dt_str, model[:-4])
    #filename = "animations/{}_radar_animation_{}_{}_{}.mp4".format(prior, dt_str, model[:-4], model2[:-4])

    # Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
    colorbar_axes = fig.add_axes([0.15, 0.1, 0.73, 0.03])

    # Create plot frames
    with writer.saving(fig, filename, 300):
        for t in range(2, 23): #range(13): #23): #8):
            print('time = ', (t+1)*5) #15)
            ax = fig.add_subplot(1,3,1)
            cf = iplt.contourf(r_cubelist[t], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('Radar', fontsize=18)
            ax = fig.add_subplot(1,3,2)
            #pdb.set_trace()
            print(nn_cubelist[0][t+1])
            cf = iplt.contourf(nn_cubelist[0][t+1], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            cf.cmap.set_over('white')
            plt.gca().coastlines('50m', color='white')
            plt.title('ML prediction', fontsize=18)
            #ax = fig.add_subplot(1,3,3)
            #cf = iplt.contourf(n_cubelist[t], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            #cf = iplt.contourf(nn_cubelist2[0][t+1], levels, colors=colors, origin='lower', extend='max') #, cmap=cm) #, vmin=0, vmax=16)
            #cf.cmap.set_over('white')
            #plt.gca().coastlines('50m', color='white')
            #plt.title('Op nowcast', fontsize=18)
            #plt.title('SVG', fontsize=15)

            # Add the colour bar
            cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            cbar.ax.set_xlabel('Rain rate (mm/hr)')

            fig.suptitle('T+{:02d} min'.format((t+1)*5), fontsize=30)
            # Save frame
            writer.grab_frame()
    plt.close()

def load_nn_pred(dt_str, model, domain, plotdi):
    nn_cubelist = []
    nn_f = '{}/plots_nn_T{}_{}.nc'.format(plotdi, dt_str, model[:-4])
    #nn_f = '/data/cr1/cbarth/phd/SVG/test.nc'
    #if (os.path.isfile(nn_f)==False):
    #     nn_f = '/data/cr1/cbarth/phd/SVG/model_output/model624800/plots_nn_T{}_{}.nc'.format(dt_str, model[:-4])
    print(nn_f)
    # Load netcdf file, avoiding the TypeError: unhashable type: 'MaskedConstant'
    cube_gen = iris.fileformats.netcdf.load_cubes(nn_f)
    nn_cubes = list(cube_gen)
    nn_cube = nn_cubes[0] #*2
    #nn_cube = nn_cube[:, 25:-25, 25:-25] #when running with reduced domain
    print(nn_cube)
    for t in range(1, 25): #[3, 6, 9, 12, 15, 18, 21, 24]:
        nn_cubelist.append(nn_cube)

    return nn_cubelist

def load_nowcast(dt_str, sample_points, domain):
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
    nowcast_cube = nc_cube[:, domain[0]:domain[1], domain[2]:domain[3]]/32
    for cu in [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]: #range(8): #[1, 3]: #, 5]: #i.e. t+30, t+60, t+90
        n_cubelist.append(nowcast_cube[cu])

    return n_cubelist #nowcast_cube #n_cubelist

def load_radar(dt, dt_str, sample_points, domain, datadi):
    # Load radar data
    r_cubelist = []
    for t in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]: #[15, 30, 45, 60, 75, 90, 105, 120]
        ti = (dt + timedelta(minutes = t)).strftime('%Y%m%d%H%M')
        radar_f = '{}/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(datadi, ti)
        radar = iris.load(radar_f)
        r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
        radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]]/32
        r_cubelist.append(radar_cube)
        print(radar_cube)
        #pdb.set_trace()

    return r_cubelist

if __name__ == "__main__":
    main()
