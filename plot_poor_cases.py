from datetime import datetime, timedelta
import iris
import csv
import nimrod_to_cubes as n2c
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import iris.plot as iplt

fig = plt.figure()
# Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
colorbar_axes = fig.add_axes([0.15, 0.1, 0.73, 0.03])
domain = [160, 288, 130, 258]
colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]
sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                 ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]
with open("poor_cases.csv") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader): # each row is a list
        if i == 0:
            headers = row
        else:
            plt.close()
            #pdb.set_trace()
            ax = fig.add_subplot(1,1,1)
            dt_str = row[1]
            print(dt_str)
            radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
            radar = iris.load(radar_f)
            #pdb.set_trace()
            r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
            radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]]/32
            cf = iplt.contourf(radar_cube, levels, colors=colors, origin='lower', extend='max')
            plt.gca().coastlines('50m', color='white')
            cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
            cbar.ax.set_xlabel('Rain rate (mm/hr)')
            plt.savefig('poor_cases/test{}.png'.format(dt_str))
            #plt.show()
            pdb.set_trace()
