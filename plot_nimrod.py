from datetime import datetime, timedelta
import iris
import nimrod_to_cubes as n2c
import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import iris.quickplot as qplt

dt_str = '201909291300' #201909201700' #201909281510' #201909291300' #201909291715' #1545' #201910251230' #271115'
#nwcst_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
radar_f = '/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str)
sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                 ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]
domain = [160, 288, 130, 258]
# cubelist = n2c.nimrod_to_cubes(nwcst_f)
#
# radar = cubelist[0] #2]
# print(radar)
# r_cube = radar.interpolate(sample_points, iris.analysis.Linear())
# radar_cube = r_cube[160:288, 130:258]/32 #:, 160:288, 130:258]/32
# #pdb.set_trace()
# qplt.contourf(radar_cube) #[0])
# plt.show()

# radar = iris.load(radar_f)
# r_cube = radar[0].interpolate(sample_points, iris.analysis.Linear())
# radar_cube = r_cube[domain[0]:domain[1], domain[2]:domain[3]]/32
# qplt.contourf(radar_cube) #[0])
# plt.show()
#
# dt_str = '201909291300' #201909201700' #201909291200' #201909291545'
# nn_f = '/data/cr1/cbarth/phd/SVG/plots_nn_T{}_model1734435.nc'.format(dt_str) #nn_T201910251230.nc'
# cube2 = iris.fileformats.netcdf.load_cubes(nn_f)
# cube = list(cube2)[0]
# print(cube[0])
# qplt.contourf(cube[0]) #*32
# plt.show()

#op_n_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast/{}_u1096_ng_pp_precip_2km'.format(dt_str)
op_n_f = '/data/cr1/cbarth/phd/SVG/verification_data/op_nowcast_5min/201901080745_u1096_ng_pp_precip5min_2km' #{}_u1096_ng_pp_precip5min_2km'.format(dt_str)

cubelist = n2c.nimrod_to_cubes(op_n_f)
pdb.set_trace()
n_cubelist = []
nowcast = cubelist[2]
nc_cube = nowcast.interpolate(sample_points, iris.analysis.Linear())
nowcast_cube = nc_cube[:, domain[0]:domain[1], domain[2]:domain[3]] / 32
print(nowcast_cube[0])
pdb.set_trace()
qplt.contourf(nowcast_cube[0])
plt.show()

pdb.set_trace()
