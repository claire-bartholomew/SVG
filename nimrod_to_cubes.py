# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************
"""This code is designed as an add-on to Iris to read Nimrod-format files and construct Iris cubes
   containing useful and CF-compliant meta-data where possible."""

import iris
import iris.fileformats.nimrod as nf
import datetime as dt
import numpy as np
import warnings
from struct import unpack
import os
import struct
import re

class NimrodException(Exception):
    """Local exception for failures handling Nimrod files"""
    pass


def check_version(module, acceptable_version):
    """Returns True if __version__ of module is newer than the acceptable_version list specified"""
    moduleversion = [x for x in module.__version__.split(".")]
    for i in range(len(moduleversion)):
        try:
            moduleversion[i] = int(moduleversion[i])
        except ValueError:
            pass
    versionnewer = True
    for i in range(len(acceptable_version)):
        if moduleversion[i] < acceptable_version[i]:
            versionnewer = False
            break
        elif moduleversion[i] > acceptable_version[i]:
            break
    return versionnewer

if check_version(iris, [1, 9, 0]):
    import cf_units
else:
    import iris.unit as cf_units

# Reference for Airy_1830 and international_1924 ellipsoids:
# http://fcm9/projects/PostProc/wiki/PostProcDocDomains#ProjectionConstants
# Reference for GRS80:
Airy_1830 = {'semi_major_axis': 6377563.396, 'semi_minor_axis': 6356256.910}
International_1924 = {'semi_major_axis': 6378388.000, 'semi_minor_axis': 6356911.946}
default_units = {817: 'm s^-1',
                 422: 'min^-1',
                 218: 'cm',
                 155: 'm',
                 101: 'm',
                 63: 'mm hr^-1',
                 61: 'mm',
                 58: 'Celsius',
                 12: 'mb'}


def nimrod_to_cubes(filename, quiet=True, ignoreErrors=False, metaonly=False, debug=False, printheader=False):
    '''
    Given the path to a Nimrod file, reads the file and attempts to build a
    list of cubes (CubeList) with some sensible metadata.

    Returns the list of cubes and a set of field codes in the file.
    ignoreErrors=True prevents code aborting if an incomplete field is found
                      (failed field and all subsequent fields will be excluded)
    quiet=True suppresses the summary printout
    metaonly=True suppresses the data-load and returns cubes with no data in them.
    printheader=True verbosely prints the count, VT, DT and IH(31) from the headers.
    '''
    # create a Nimrod file object
    nimrodobj = nf.NimrodField()
    count = 0
    cnt = 0
    cubes = iris.cube.CubeList()
    time_unit = cf_units.Unit('hours since 1970-01-01 00:00:00')
    header_data = HeaderInfo(filename, quiet=quiet, ignoreErrors=ignoreErrors, debug=debug)
    with open(filename, "rb") as infile:
        while True:
            try:
                nimrodobj.read(infile)
                if nimrodobj.vt_year <= 0:
                    # Some ancillary files, eg land sea mask do not
                    # have a validity time. So make one up for the
                    # start of the year.
                    # This will screw up the forecast_period for these fields,
                    # although if the valid time is missing too, it will be
                    # made to be the same, so the forecast_period will always
                    # be zero for these files.
                    valid_time = dt.datetime(
                        2016, 1, 1, 0, 0, 0)
                else:
                    valid_time = dt.datetime(
                        nimrodobj.vt_year, nimrodobj.vt_month, nimrodobj.vt_day,
                        nimrodobj.vt_hour, nimrodobj.vt_minute, 0)
                # Handle case where data time is set to missing (ancillary
                # files)
                if nimrodobj.dt_year == -32767 or nimrodobj.dt_year == 0:
                    data_time = valid_time
                else:
                    data_time = dt.datetime(
                        nimrodobj.dt_year, nimrodobj.dt_month,
                        nimrodobj.dt_day, nimrodobj.dt_hour,
                        nimrodobj.dt_minute, 0)

                if printheader:
                    print(count, valid_time, data_time, header_data.averagingtype[count])

                # Get size of x, y data and the x, y origins from header.
                nx = header_data.colnumbers[count]
                ny = header_data.rownumbers[count]
                if metaonly:
                    nx = 2
                    ny = 2
                xorigin = header_data.x_origins[count]
                yorigin = header_data.y_origins[count]
                lat_true_orig = header_data.lat_true_orig[count]
                long_true_orig = header_data.long_true_orig[count]
                easting_true_orig = header_data.easting_true_orig[count]
                northing_true_orig = header_data.northing_true_orig[count]
                central_meridian_sf = header_data.central_meridian_sf[count]
                x_res = header_data.x_resol[count]
                y_res = header_data.y_resol[count]
                ellipsoid = header_data.ellipsoid[count]
                imdi = header_data.imdi[count]
                grid = header_data.gridtype[count]
                # Set up x and y coordinates based on header information.
                x_coord, y_coord = _add_horizontal_coords(nx, ny,
                                                          xorigin, yorigin,
                                                          lat_true_orig,
                                                          long_true_orig,
                                                          easting_true_orig,
                                                          northing_true_orig,
                                                          central_meridian_sf,
                                                          x_res, y_res,
                                                          grid, ellipsoid,
                                                          imdi)

                nimrodobj, threshold_coord = _updatetitle(nimrodobj, header_data, count)
                if metaonly:
                    thisdata = np.zeros([ny, nx])
                else:
                    if header_data.datatype[count] == 1:
                        thisdata = np.ma.masked_equal(nimrodobj.data, header_data.imdi[count])
                    elif header_data.datatype[count] == 0:
                        thisdata = np.ma.masked_inside(nimrodobj.data,
                                                       header_data.rmdi[count]-0.5,
                                                       header_data.rmdi[count]+0.5)
                    else:
                        thisdata = np.zeros([ny, nx])
                try:
                    cube = iris.cube.Cube(thisdata,
                                          long_name=nimrodobj.title.strip(),
                                          units=None,
                                          dim_coords_and_dims=[(y_coord, 0),
                                                               (x_coord, 1)])
                except:
                    raise NimrodException(
                        'Unable to create basic cube with data, x and y coords')

                ## Add units to cube
                #unitstring = header_data.units[count]
                #try:
                #    _add_units(unitstring, cube)
                #except:
                #    raise NimrodException(
                #        'Units not recognised by Iris: ' + str(unitstring))

                # add threshold metadata to cube if present
                if threshold_coord is not None:
                    cube.add_aux_coord(threshold_coord)

                # add time and other metadata to cube
                aver_per = header_data.averaging_period[count]
                aver_units = header_data.averaging_period_units[count]
                try:
                    _add_time_coords(cube, valid_time, data_time, time_unit, aver_per, aver_units,
                                     debug=debug)
                except:
                    raise NimrodException('''Unable to add time coordinates for time {} and
                                             forecast_reference_time
                                             {}'''.format(valid_time, data_time))

                # Add vertical coordinate to cube
                height_val = header_data.vert_value[count]
                height_type = header_data.vert_coord_type[count]
                try:
                    _add_vertical_coord(cube, height_type, height_val,
                                        header_data.vertupper_value[count])
                except:
                    raise NimrodException('''Height metadata issue, unable to add height
                                             coordinate to cube, height_type:{}, height value:
                                             {}'''.format(height_type, height_val))

                # Add data source, not directly from header.
                try:
                    thissource = nimrodobj.source.strip()
                    rematcher = re.compile('^ek\d\d$')
                    if rematcher.match(thissource) is not None or thissource.find('umek') == 0:
                        thissource = 'MOGREPS-UK'
                    cube.attributes['source'] = thissource
                except:
                    raise NimrodException('Unable to add data source to cube')

                # Add probability meta-data if relevant
                multimemberfield = _add_probability_info(cube,
                                                         header_data,
                                                         count,
                                                         metaonly=metaonly)

                ## Add ensemble member if relevant
                #if nimrodobj.ensemble_member >= 0 and not multimemberfield:
                #    cube.add_aux_coord(iris.coords.AuxCoord(
                #        nimrodobj.ensemble_member, standard_name='realization',
                #        units=None))
                #if (nimrodobj.ensemble_member == -98 or
                #    any([True for x in header_data.chead[count].split(' ') if x.find('Mean') >= 0])):
                #    cube.add_cell_method(
                #        iris.coords.CellMethod('mean',
                #                               coords='realization',
                #                               intervals='{} members'.format(
                #                                   header_data.prob_nmembers[count])
                #                               ))
                #if (nimrodobj.ensemble_member == -99 or
                #    any([True for x in header_data.chead[count].split(' ') if x.find('Spread') >= 0])):
                #    cube.add_cell_method(
                #        iris.coords.CellMethod('standard_deviation',
                #                               coords='realization',
                #                               intervals='{} members'.format(
                #                                   header_data.prob_nmembers[count])
                #                               ))

                # Add soil type if relevant
                if header_data.thresh_type[count] != 0:
                    if header_data.soil_type[count] != -32767:
                        cube.add_aux_coord(iris.coords.AuxCoord(
                            _soiltypestring(header_data.soil_type[count]),
                            standard_name='soil_type', units=None))
                # Add relevant cell methods using header information.
                # Add cell method 'None' if no cell methods so can still
                # search by cell method.

                # Add field number as an attribute to the cube
                cube.attributes.update(
                    {"field_code": header_data.field_numbers[count]})
                cell_methods, attributes = _get_averaging(
                    header_data.averagingtype[count], str(aver_per) + ' ' + aver_units)
                for method in cell_methods:
                    cube.add_cell_method(method)
                if len(attributes) > 0:
                    cube.attributes.update({'processing': attributes})
                # Add experiment_no
                experiment_no = header_data.experiment_no[count]
                cube.attributes.update({'experiment_number': experiment_no})
                # add cube to cubelist
                cubes.append(cube)
                count += 1
                cnt += 1
            except NimrodException as E:
                count += 1
                print('ERROR: Could not read field number', count, ':', E)
            except struct.error:
                if not quiet:
                    print('Number of fields in file:', len(header_data.field_numbers))
                    print('Reached end of file:', count, 'field(s) read')

                if cnt != len(header_data.field_numbers):
                    warnings.warn('Not all data loaded into cube list')
                break
            except Exception as E:
                if ignoreErrors:
                    if not quiet:
                        print('ERROR: Could not read field number', count, ':', E)
                    warnings.warn('Not all data loaded into cube list')
                    break
                else:
                    print('ERROR: Could not read field number', count, ':', E)
                raise
    if not quiet:
        print('Field codes in file: ', set(header_data.field_numbers))
    return cubes.merge(unique=False)


def _add_horizontal_coords(nx, ny, xorigin, yorigin, lat_true_orig,
                           long_true_orig, easting_true_orig,
                           northing_true_orig, central_meridian_sf,
                           x_res, y_res, grid, ellipsoid, idmi):
    '''
    Returns a tuple containing instances of iris.coords.DimCoord() for the x
    and y coordinates.

    Creates suitable x and y coordinates based on the information taken from
    the header information. Currently only National Grid, Latitiude/longitude
    and EuroPP domains are supported (values 0,1,4 from Nimrod header entry 15,
    horizontal grid type).

    Args:
        * nx - the number of points in the x direction
        * ny - the number of points in the y direction
        * xorigin - the easting or longitude of first x point (in m or degrees)
        * yorigin - the northing or latitude of first y point (in m or degrees)
        * lat_true_orig - standard latitude of true origin in degrees
        * long_true_orig - standard longitude of true origin in degrees
        * easting_true_orig - Easting of true origin in m
        * northing_true_orig - Northing of true origin in m
        * central_meridian_sf - scale facetor on the central meridian
        * x_res - the resolution of the x points (in m or degrees)
        * y_res - the resolution of the y points (in m or degrees)
        * grid - the horizontal grid type taken from Nimrod header entry 15
        * ellipsoid - the projection biaxial ellipsoid taken from Nimrod header
                    entry 28
        * idmi - the missing data value taken from Nimrod header entry 25
    '''

    ellipsoid = _get_ellipsoid(ellipsoid, grid, idmi)

    if grid == 0 or grid == 4:
        if central_meridian_sf == idmi and grid == 4:  # some e3195 files miss
            central_meridian_sf = 0.9996  # this, number from NIMRODdoc grid=4
        coord_syst = iris.coord_systems.TransverseMercator(
            lat_true_orig, long_true_orig, easting_true_orig, northing_true_orig,
            central_meridian_sf, iris.coord_systems.GeogCS(**ellipsoid))
        units = 'm'
        x_coord_name = 'projection_x_coordinate'
        y_coord_name = 'projection_y_coordinate'
    elif grid == 1:
        coord_syst = iris.coord_systems.GeogCS(**ellipsoid)
        units = 'degrees'
        x_coord_name = 'longitude'
        y_coord_name = 'latitude'
    else:
        raise NimrodException('Unsupported grid type, only NG, EuroPP and lat/long are possible')

    x_points = xorigin + np.arange(nx) * x_res
    y_points = yorigin - np.arange(ny) * y_res
    x_coord = iris.coords.DimCoord(x_points, standard_name=x_coord_name,
                                   units=units, coord_system=coord_syst,
                                   circular=False,)
    x_coord.guess_bounds()
    y_coord = iris.coords.DimCoord(y_points, standard_name=y_coord_name,
                                   units=units, coord_system=coord_syst,
                                   circular=False,)
    y_coord.guess_bounds()
    return x_coord, y_coord


def _get_ellipsoid(ellipsoid, grid, idmi):
    '''
    Returns the correct dictionary of arguements needed to define an
    iris.coord_systems.GeogCS.

    Based firstly on the value given by ellipsoid, then by grid if ellipsoid is
    missing, select the right pre-defined ellipsoid dictionary (Airy_1830 or
    International_1924).

    Args:
        * ellipsoid - the projection biaxial ellipsoid taken from Nimrod header
                    entry 28
        * grid - the horizontal grid type taken from Nimrod header entry 15
        * idmi - the missing data value taken from Nimrod header entry 25
                 Also checks default missing data value of -32767

    '''
    if ellipsoid == 0:
        ellipsoid = Airy_1830
    elif ellipsoid == 1:
        ellipsoid = International_1924
    elif ellipsoid == idmi or ellipsoid == -32767:
        if grid == 0:
            ellipsoid = Airy_1830
        elif grid == 1 or grid == 4:
            ellipsoid = International_1924
        else:
            raise NimrodException('''Unsupported grid type, only NG, EuroPP
                                     and lat/long are possible''')
    else:
        raise NimrodException('Ellipsoid not supported, ellipsoid:{}, grid:{}'.format(ellipsoid,
                                                                                      grid))
    return ellipsoid


def _updatetitle(nimrodobj, header_data, count):
    """Attempts to modify a Nimrod object title based on other meta-data in the Nimrod field."""
    nimobjloc = nimrodobj
    thresh_sn = None
    if header_data.field_numbers[count] == 161 and header_data.thresh_value[count] >= 0.:
        nimrodobj.title = "minimum_cloud_base_above_thresho"
        thresh_sn = "cloud_area_fraction"
        thresh_units = ""
        thresh_value = header_data.thresh_value[count]
    if header_data.field_numbers[count] == 12:
        nimrodobj.title = "pressure"
    if header_data.field_numbers[count] == 28:
        nimrodobj.title = "snow probability"
    if header_data.field_numbers[count] == 29 and header_data.thresh_value[count] >= 0.:
        nimrodobj.title = "fog fraction"
        thresh_sn = "visibility_in_air"
        thresh_units = "metres"
        thresh_value = header_data.thresh_value[count]
    if header_data.field_numbers[count] == 58:
        nimrodobj.title = "temperature"
    if header_data.field_numbers[count] == 61:
        nimrodobj.title = "precipitation"
    if header_data.field_numbers[count] == 63:
        nimrodobj.title = "precipitation"
    if header_data.field_numbers[count] == 817:
        nimrodobj.title = "wind_speed_of_gust"
    if header_data.field_numbers[count] == 155:
        nimrodobj.title = "Visibility"
    if header_data.field_numbers[count] == 218:
        nimrodobj.title = "snowfall"
    if header_data.field_numbers[count] == 101:
        nimrodobj.title = "snowmelt_above_sea_level"
    if header_data.field_numbers[count] == 172:
        nimrodobj.title = "cloud_area_fraction_in_atmospher"
        thresh_sn = None
    if header_data.field_numbers[count] == 421:
        nimrodobj.title = "precipitation type"
    if header_data.field_numbers[count] == 804 and header_data.vert_value[count] >= 0.:
        nimrodobj.title = "wind speed"
        thresh_sn = None
    if header_data.field_numbers[count] == 806 and header_data.vert_value[count] >= 0.:
        nimrodobj.title = "wind direction"
        thresh_sn = None
    if header_data.field_numbers[count] == 821:
        nimrodobj.title = "Probabilistic Gust Risk Analysis from Observations"
        nimrodobj.source = "Nimrod pwind routine"
        thresh_sn = "wind_speed_of_gust"
        thresh_units = "m/s"
        thresh_value = header_data.thresh_value[count]
    if nimrodobj.source.strip() == "pwind":
        nimrodobj.source = "Nimrod pwind routine"
    if thresh_sn is None:
        return nimobjloc, None
    else:
        return nimobjloc, iris.coords.AuxCoord(thresh_value,
                                               standard_name=thresh_sn,
                                               units=thresh_units)


def _add_time_coords(cube, time_value, forecast_reference_time, time_unit, bound_length,
                     bound_units, debug=False):
    '''
    Adds time coordinates to the cube.

    Adds a 'time' coordinate representing the Validity time for the data.
    Adds a 'forecast_reference_time' coordinate representing the data time for
    the data.
    Adds a 'forecast_period' coordinate representing the difference between the
    'forecast_period' and the 'time' coordinate in seconds.

    Args:
     * cube (Cube) - the cube we want to add units to.
     * time_value (datetime object) - the datetime object for the time
            coordinate
     * forecast_reference_time (datetime object) - the datetime object for the
            cube's forecast reference time
     * time_unit (a cf_units.Unit() instance ) - the units for the all the
            time axes.
     * bound_length (number) - length of relevant time-window for coords bounds
     * bound_units (string) - units for bound_length

    '''
    time_value_num = time_unit.date2num(time_value)
    forecast_reference_time_num = time_unit.date2num(forecast_reference_time)
    lb_delta = None
    if debug:
        print('Attempting to add bounds starting {} {} before validity-time.'.format(bound_length,
                                                                                     bound_units))
    if bound_length > 0:
        if  (bound_units is 'minutes' or
             bound_units is 'minute' or
             bound_units is 'mins' or
             bound_units is 'min'):
            lb_delta = dt.timedelta(minutes=bound_length)
        if  (bound_units is 'hours' or
             bound_units is 'hour' or
             bound_units is 'hrs' or
             bound_units is 'hr'):
            lb_delta = dt.timedelta(hours=bound_length)
        if  (bound_units is 'seconds' or
             bound_units is 'second' or
             bound_units is 'sec'):
            lb_delta = dt.timedelta(seconds=bound_length)
    if lb_delta is not None:
        bounds = [time_unit.date2num(time_value - lb_delta), time_value_num]
    else:
        bounds = None
    cube.add_aux_coord(iris.coords.AuxCoord(time_value_num,
                                            standard_name='time',
                                            bounds=bounds,
                                            units=time_unit))
    cube.add_aux_coord(iris.coords.AuxCoord(
        forecast_reference_time_num, standard_name='forecast_reference_time',
        units=time_unit))
    forecast_period = time_value - forecast_reference_time
    # forecast_period object only has days and seconds attributes but the
    # seconds are often >360, ie it cover several hours.
    seconds = forecast_period.days * 24 * 60 * 60 + forecast_period.seconds
    u = cf_units.Unit('second')
    if lb_delta is not None:
        bounds = [seconds - lb_delta.total_seconds(), seconds]
    else:
        bounds = None
    if debug:
        print('Applying bounds of {}'.format(bounds))
    cube.add_aux_coord(
        iris.coords.AuxCoord(seconds, long_name='forecast_period', bounds=bounds, units=u))


def _add_units(unitstring, cube):
    '''
    Adds units to the cube.

    Takes into account nimrod unit strings of the form unit*?? where the data
    needs to converted by dividing by ??. Also converts units we know Iris
    can't handle into appropriate units Iris can handle. This is mostly when
    there is an inappropriate capital letter in the unit in the Nimrod file.
    Some units still can't be handled by Iris so in these cases empty strings
    are added as the cube's unit. The most notable unit Iris can't handle is
    oktas for cloud cover.

    Args:
     * unitstring (string) - the string relating to the data's units retrieved
            from the file header.
     * cube (Cube) - the cube we want to add units to.
    '''
    unitstring = unitstring.strip('\x00')

    unit_exception_dictionary = {'Knts': 'knots',
                                 'knts': 'knots',
                                 'J/Kg': 'J/kg',
                                 'logical': '',
                                 'Code': '',
                                 'mask': '',
                                 'oktas': '',
                                 'm/2-25k': '',
                                 'g/Kg': '',
                                 'unitless': '',
                                 'Fraction': '1',
                                 'index': '',
                                 'Beaufort': '',
                                 'mmh2o': 'kg/m2', }
    if '*' in unitstring:
        # Split into unit string and integer
        unit_list = unitstring.split('*')
        cube.data = cube.data / float(unit_list[1])
        unitstring = unit_list[0]
    if 'ug/m3E1' in unitstring:
        # Split into unit string and integer
        unit_list = unitstring.split('E')
        cube.data = cube.data / 10.**float(unit_list[1])
        unitstring = unit_list[0]
    unitstring = unitstring.strip()
    if '%' in unitstring:
        # Convert any percentages into fraction
        unit_list = unitstring.split('%')
        if len(''.join(unit_list)) == 0:
            unitstring = '1'
            cube.data = cube.data / 100.
    if unitstring in unit_exception_dictionary.keys():
        unitstring = unit_exception_dictionary[unitstring]
    if len(unitstring) > 0 and unitstring[0] == '/':
        # Deal with the case where the units are of the form '/unit' eg
        # '/second' in the Nimrod file. This converts to the form unit^-1
        unitstring = unitstring[1:] + '^-1'
    # Update cube units
    cube.units = unitstring


def _get_averaging(num, period=None):
    '''
    Returns a list of cell_methods and a list of attributes which describe the
    processing that has been applied to the data.

    Args:
     * num (integer) - the integer from Nimrod header entry 31, Time averaging
                       combinations (LBPROC)
     * period (string) - description of the time-period that applies, e.g. '1 hour'
    '''

    cell_methods = []
    attributes = []
    time_averaging_dictionary = {8192: {'cf_name': 'maximum', 'cf_coord': 'time',
                                        'long_name': "maximum in period", 'cell': True},
                                 4096: {'cf_name': 'minimum', 'cf_coord': 'time',
                                        'long_name': 'minimum in period', 'cell': True},
                                 2048: {'long_name': 'unknown(2048)', 'cell': False},
                                 1024: {'long_name': 'unknown(1024)', 'cell': False},
                                 512: {'long_name': 'time lagged', 'cell': False},
                                 256: {'long_name': 'extrapolation', 'cell': False},
                                 128: {'cf_name': 'mean', 'cf_coord': 'time',
                                       'long_name': 'accumulation or average', 'cell': True},
                                 64: {'long_name': 'from UM 150m', 'cell': False},
                                 32: {'long_name': 'scaled to UM resolution', 'cell': False},
                                 16: {'cf_name': 'mean', 'cf_coord': 'soil_type',
                                      'long_name': 'averaged over multiple surface types',
                                      'cell': True},
                                 8: {'long_name': 'only observations used', 'cell': False},
                                 4: {'long_name': 'smoothed', 'cell': False},
                                 2: {'long_name': 'cold bias applied', 'cell': False},
                                 1: {'long_name': 'warm bias applied', 'cell': False}, }

    for key in sorted(time_averaging_dictionary.keys(), reverse=True):
        if num >= key:
            thismethod = time_averaging_dictionary[key]
            if thismethod['cell']:
                cell_methods.append(iris.coords.CellMethod(thismethod['cf_name'],
                                                           coords=thismethod['cf_coord'],
                                                           comments=thismethod['long_name'],
                                                           intervals=period))
            else:
                attributes.append(thismethod['long_name'])
            num = num - key
    if len(cell_methods) == 0:  # Add default cell_method.
        cell_methods.append(iris.coords.CellMethod('point', coords='time'))
    return cell_methods, attributes


def _soiltypestring(num):
    """Returns human-readable string from soil-type ID number"""
    if num == 1:
        string = "Broadleaf Tree"
    elif num == 2:
        string = "Needleleaf Tree"
    elif num == 3:
        string = "C3 Grass"
    elif num == 4:
        string = "C4 Grass"
    elif num == 5:
        string = "Crop"
    elif num == 6:
        string = "Shrub"
    elif num == 7:
        string = "Urban"
    elif num == 8:
        string = "Water"
    elif num == 9:
        string = "Soil"
    elif num == 10:
        string = "Ice"
    elif num == 601:
        string = "Urban Canyon"
    elif num == 602:
        string = "Urban Roof"
    else:
        string = "Unknown"
    return string


def _add_probability_info(cube, header_data, count, metaonly=False, debug=False):
    '''
    Adds probability meta-data from the header to the cube
    '''
    probtype_lookup = {1: 'Probability above',
                       2: 'Probability below',
                       3: 'Percentile',
                       4: 'Probability equal'}
    probmethod_lookup = {1: 'AOT (Any One Time)',
                         2: 'ST (Some Time)',
                         4: 'AT (All Time)',
                         8: 'AOL (Any One Location)',
                         16: 'SW (Some Where)'}
    multimemberfield = False
    try:
        probname = probtype_lookup[header_data.thresh_type[count]]
    except KeyError:
        if debug:
            print("""Not a probability field as thresh_type ({}) not
                  matched""".format(header_data.thresh_type[count]))
        return multimemberfield
    if header_data.thresh_type[count] == 3:
        units = "%"
    else:
        try:
            units = default_units[header_data.field_numbers[count]]
        except KeyError:
            units = None
            if debug:
                print("No default units for field code {}".format(header_data.field_numbers[count]))
    thisval = None
    if header_data.thresh_value_alt[count] > -32766.:
        thisval = header_data.thresh_value_alt[count]
    elif header_data.thresh_value[count] > -32766.:
        thisval = header_data.thresh_value[count]
    if header_data.chead[count].find('pc') > 0:
        try:
            thisval = [int(x.strip('pc')) for x in header_data.chead[count].split(' ') if x.find('pc') > 0][0]
        except IndexError:
            pass
    if thisval is not None:
        if header_data.prob_fuzzyth[count] > -32766.:
            bounds = [thisval * header_data.prob_fuzzyth[count],
                      thisval * (2. - header_data.prob_fuzzyth[count])]
        else:
            bounds = None
        new_coord = iris.coords.AuxCoord(thisval, standard_name=None,
                                         long_name=probname, units=units, bounds=bounds)
        if debug:
            print("Adding probability coordinate")
            print(new_coord)
        cube.add_aux_coord(new_coord)
        cube.units = '1'
        multimemberfield = True
    else:
        if debug:
            print("""Not a probability field as thresh_value ({}) and thresh_value_alt ({}) not
                  valid""".format(header_data.thresh_value[count],
                                  header_data.thresh_value_alt[count]))

    try:
        xdim_points = cube.coord('projection_x_coordinate')
    except iris.exceptions.CoordinateNotFoundError:
        xdim_points = cube.coord('longitude')
    if header_data.prob_neighradius[count] > 0:
        if metaonly:
            neigh_radius = '{} {}'.format((xdim_points.points[1] - xdim_points.points[0])
                                          * header_data.prob_neighradius[count],
                                          xdim_points.units)
        else:
            radval = header_data.prob_neighradius[count]
            try:
                neigh_radius = xdim_points[radval] - xdim_points[0]
            except iris.exceptions.NotYetImplementedError:
                neigh_radius = '{} {}'.format(xdim_points.points[radval] - xdim_points.points[0],
                                              xdim_points.units)
        if header_data.prob_niters[count] >= 0 and 0. <= header_data.prob_rfalpha[count] <= 1.:
            commentstr = """Neighbourhood + {}.RecursiveFilter ({})""".format(header_data.prob_niters[count],
                                           header_data.prob_rfalpha[count])
        else:
            commentstr = 'Neighbourhood'
        cube.add_cell_method(iris.coords.CellMethod('mean',
                                                    coords='area',
                                                    comments=commentstr,
                                                    intervals=neigh_radius))
    if header_data.prob_radius[count] > 0:
        if metaonly:
            radius = '{} {}'.format((xdim_points.points[1] - xdim_points.points[0])
                                    * header_data.prob_radius[count],
                                    xdim_points.units)
        else:
            radval = header_data.prob_radius[count]
            try:
                radius = xdim_points[radval] - xdim_points[0]
            except iris.exceptions.NotYetImplementedError:
                radius = '{} {}'.format(xdim_points.points[radval] - xdim_points.points[0],
                                        xdim_points.units)
        commentstr = None
        cube.add_cell_method(iris.coords.CellMethod('mean',
                                                    coords='area',
                                                    comments=commentstr,
                                                    intervals=radius))
    attributes = []
    if header_data.prob_method[count] > 0:
        num = header_data.prob_method[count]
        for key in sorted(probmethod_lookup.keys(), reverse=True):
            if num >= key:
                thismethod = probmethod_lookup[key]
                attributes.append(thismethod)
                num = num - key
        cube.attributes['Probability methods'] = attributes
    if header_data.prob_nmembers[count] >= 0:
        nummemb = header_data.prob_nmembers[count]
        if nummemb in [1, -32767]:
            multimemberfield = False
        cube.add_cell_method(iris.coords.CellMethod('frequency',
                                                    coords='realization',
                                                    intervals='{} members'.format(nummemb)))
    if header_data.prob_nmembers[count] == 1:
        multimemberfield = False
    if header_data.prob_threshwindow[count] >= 0:
        cube.attributes['Probability methods'].append("""minimum_time {} minutes""".format(header_data.prob_threshwindow[count]))
    if debug:
        print('Not used prob_fuzzytime ({})'.format(header_data.prob_fuzzytime[count]))
    return multimemberfield


def _add_vertical_coord(cube, height_type, height_val, height_val_upper):
    '''
    Adds a vertical coordinate to the cube.

    Builds a veritcal coordinate based on the height_type which is converted
    into a coordinate name with unit and the value for the vertical coordinate.

    Args:
     * cube (Cube) - the cube we want to add units to.
     * hieght_type (integer) - the header entry 20 in the Nimrod file
     * height_val (float) - the value of the height coordinate.
    '''

    # mydict contains convertions from the Nimrod Documentation for the header
    # entry 20 for the vertical coordinate type
    mydict = {0: ['Height above orography', 'm'],
              1: ['Height above sea level', 'm'],
              2: ['pressure', 'hPa'],
              3: ['sigma', 'model level'],
              4: ['eta', 'model level'],
              5: ['radar beam number', 'unknown'],
              6: ['temperature', 'K'],
              7: ['potential temperature', 'unknown'],
              8: ['equivalent potential temperature', 'unknown'],
              9: ['wet bulb potential temperature', 'unknown'],
              10: ['potential vorticity', 'unknown'],
              11: ['cloud boundary', 'unknown'],
              12: ['levels below ground', 'unknown']}
    if height_val_upper == 9999. and height_val == 9999.:
        height_val_upper = 0.  # A bit nonsensical.
    if height_val == 9999. or height_val == 8888.:
        height_val = 0.  # Relies on correct vert_coord_type instead.

    if height_val_upper >= 0. and height_val_upper != height_val:
        height_bounds = [height_val, height_val_upper]
    else:
        height_bounds = None
    try:
        long_name = mydict[height_type][0]
        units = mydict[height_type][1]
        new_coord = iris.coords.AuxCoord(height_val, standard_name='height',
                                         long_name=long_name, units=units,
                                         bounds=height_bounds)
    except KeyError:
        # If there is a height type that is unknown then it is added anyway
        # with 'unknown' units.
        units = 'unknown'
        new_coord = iris.coords.AuxCoord(
            height_val, standard_name='height', units=units)

    finally:
        # Add coordiate to cube
        cube.add_aux_coord(new_coord)


def select_cubes(cubelist, **kwargs):
    '''
    A function to search for cubes matching kwargs in a list of cubes
    and merge matching cubes into a higher dimensional cube which is
    returned.

    Valid kwargs:
      field_code = int where int is any integer which corresponds to a
        Nimrod field code.
       cell_methods=[] where the list contains any number of
        iris.coords.CellMethod() objects
      height = int or float of a valid height value for the height
        coordinate in a cube.
      fc_times = [] where the list contains some int or float forecast lead
        times (in hours).
      averaging_period=string where the string is the averaging or
        accumulation period of interest with the units, eg 60 mins.
      experiment_no=int where int is an experiment number (multiple of 4)
      attribute=string where the string labels a True attribute in the cube
        set by _get_averaging e.g. attribute='smoothed'
     To do: add ability to search for any other metadata
    '''
    # Create new cubelist to store matching cubes
    # Test to see if valid keywords have been provided
    keys = ['field_code', 'cell_methods', 'height', 'fc_times', 'averaging_period',
            'experiment_no', 'attribute']
    for key in kwargs.iterkeys():
        if key not in keys:
            raise NimrodException('Invalid keyword')
    if len(kwargs) < 1:
        raise NimrodException('No valid constraints were provided')
    # Set up Iris constraints corrsponding to supplied kwargs.
    # If fieldcode kwarg is supplied add constraint for the cube
    # field code matching that given.
    constraints = iris.Constraint()
    if 'field_code' in kwargs:
        fieldcode = kwargs['field_code']
        constraints = constraints & iris.AttributeConstraint(
            field_code=fieldcode)
    # If CellMethod kwarg is supplied add constraint for the cube
    # cell method to match those given
    if 'cell_methods' in kwargs:
        for item in kwargs['cell_methods']:
            if not isinstance(item, iris.coords.CellMethod):
                continue
            constraints = constraints & iris.Constraint(
                cube_func=lambda cube: item in [x for x in cube.cell_methods])
    # If height kwarg is supplied add constraint cube height coordinate
    # value to match that given.
    if 'height' in kwargs:
        heightval = kwargs['height']
        constraints = constraints & iris.Constraint(height=heightval)
    if 'fc_times' in kwargs:
        [cube.coord('forecast_period').convert_units('hours') for cube in cubelist]
        constraints = constraints & iris.Constraint(
            forecast_period=lambda cell: cell.point in kwargs['fc_times'])
    if 'averaging_period' in kwargs:
        averaging_period = kwargs['averaging_period']
        constraints = constraints & iris.Constraint(
            cube_func=lambda cube: averaging_period in [x.intervals[0] for x in cube.cell_methods])
    if 'experiment_no' in kwargs:
        exp_no = kwargs['experiment_no']
        constraints = constraints & iris.AttributeConstraint(
            experiment_number=exp_no)
    if 'attribute' in kwargs:
        attribute_name = kwargs['attribute']
        if isinstance(attribute_name, str):
            attribute_name = [attribute_name]  # Avoids string being handled char-by-char
        constraints = constraints & iris.AttributeConstraint(
            processing=attribute_name)
    new_cube_list = cubelist.extract(constraints)
    # Try merging cubes and return merged cubelist. (May return more than
    # one cube in a cube list which match kwargs supplied.)
    output = new_cube_list.merge()
    if len(output) == 0:
        raise IndexError('No cubes with that specification found ({})'.format(kwargs))
    if len(output) > 1:
        warnings.warn(
            'More than one cube matching search critera is formed')
    return output


class HeaderInfo:

    '''
    A class designed to store header information in arrays attached to
    a HeaderInfo object. Each file would have its own HeaderInfo object
    containing the header information for each field in the file. This
    can then be accessed when building cubes to add metadata to the cube.
    '''

    def __init__(self, filename, quiet=False, ignoreErrors=False, debug=False):
        # Initialise the class with header data for the file.

        # set initial seek point to start of file
        initial_pos = 0
        # set counter
        cnt = 0
        headerbytes = 512
        paddedheaderbytes = headerbytes + 4 * 4
        # set arrays to store header data
        self.datatype = []
        datasize = []
        self.rownumbers = []
        self.colnumbers = []
        self.field_numbers = []
        self.y_origins = []
        self.x_origins = []
        self.y_resol = []
        self.x_resol = []
        self.lat_true_orig = []
        self.long_true_orig = []
        self.northing_true_orig = []
        self.easting_true_orig = []
        self.central_meridian_sf = []
        self.units = []
        self.chead = []
        self.unit_scalefactor = []
        self.gridtype = []
        self.averagingtype = []
        self.vert_coord_type = []
        self.vert_value = []
        self.vertupper_coord_type = []
        self.vertupper_value = []
        self.averaging_period = []
        self.averaging_period_units = []
        self.experiment_no = []
        self.thresh_value = []
        self.thresh_value_alt = []
        self.thresh_type = []
        self.prob_method = []
        self.prob_niters = []
        self.prob_nmembers = []
        self.prob_threshwindow = []
        self.prob_neighradius = []
        self.prob_radius = []
        self.prob_rfalpha = []
        self.prob_fuzzyth = []
        self.prob_fuzzytime = []
        self.soil_type = []
        self.imdi = []
        self.rmdi = []
        self.ellipsoid = []

        with open(filename, 'rb') as f:
            filesize = os.path.getsize(filename)

            # get header info for each field in file.
            # initial_pos set to start point of each field
            while initial_pos < filesize:
                try:
                    cnt = cnt + 1

                    f.seek(initial_pos)
                    _pad1 = f.read(4)  # padding byte
                    # MMF: >:big endian; L:unsigned long
                    _pad1 = unpack('>L', _pad1)
                    if _pad1[0] != 512:
                        raise NimrodException("""Nimrod header seems not to be of size 512.
                                                  Something went out of phase""")

                    # might be unnecessary data here, currently not used.
                    _unpack_data(f, 'data_pos', initial_pos, 2, '>h', (self.datatype))
                    _unpack_data(f, 'byte_pos', initial_pos, 2, '>h', (datasize))
                    # Get header info for use in nimrod_to_cubes:
                    _unpack_data(f, 'row_pos', initial_pos, 2, '>h', (self.rownumbers))
                    _unpack_data(f, 'col_pos', initial_pos, 2, '>h', (self.colnumbers))
                    _unpack_data(f, 'grid_pos', initial_pos, 2, '>h', self.gridtype)
                    # if self.gridtype[
                    #        cnt - 1] != 0 and self.gridtype[cnt - 1] != 4:
                    #    print(self.gridtype)
                    #    raise ValueError("grid type currently unsupported")

                    _unpack_data(f, 'aver_period_pos', initial_pos, 2, '>h', self.averaging_period)
                    if self.averaging_period != 32767:
                        self.averaging_period_units.append('mins')
                    else:
                        self.averaging_period.pop()
                        _unpack_data(f, 'field_pos', initial_pos, 2, '>h', self.averaging_period)
                        self.averaging_period_units.append('secs')

                    _unpack_data(f, 'field_pos', initial_pos, 2, '>h', (self.field_numbers))
                    _unpack_data(f, 'vert_coord_type_pos', initial_pos, 2, '>h',
                                 (self.vert_coord_type))
                    _unpack_data(f, 'vert_value_pos', initial_pos, 4, '>f', (self.vert_value))
                    _unpack_data(f, 'vertupper_coord_type_pos', initial_pos, 2, '>h',
                                 (self.vertupper_coord_type))
                    _unpack_data(f, 'vertupper_value_pos', initial_pos, 4, '>f',
                                 (self.vertupper_value))
                    _unpack_data(f, 'averaging_pos', initial_pos, 2, '>h', (self.averagingtype))
                    _unpack_data(f, 'yorigin_pos', initial_pos, 4, '>f', (self.y_origins))
                    _unpack_data(f, 'yresol_pos', initial_pos, 4, '>f', (self.y_resol))
                    _unpack_data(f, 'xorigin_pos', initial_pos, 4, '>f', (self.x_origins))
                    _unpack_data(f, 'xresol_pos', initial_pos, 4, '>f', (self.x_resol))
                    _unpack_data(f, 'unit_scalefactor_pos', initial_pos, 4, '>f',
                                 (self.unit_scalefactor))
                    _unpack_data(f, 'unit_pos', initial_pos, 8, '>8s', (self.units))
                    _unpack_data(f, 'chead_pos', initial_pos, 48, '>48s', (self.chead))
                    _unpack_data(f, 'imdi_pos', initial_pos, 2, '>h', (self.imdi))
                    _unpack_data(f, 'rmdi_pos', initial_pos, 4, '>f', (self.rmdi))
                    _unpack_data(f, 'long_true_orig_pos', initial_pos, 4, '>f', (self.long_true_orig))
                    _unpack_data(f, 'lat_true_orig_pos', initial_pos, 4, '>f', (self.lat_true_orig))
                    _unpack_data(f, 'northing_true_orig_pos', initial_pos, 4, '>f',
                                 (self.northing_true_orig))
                    _unpack_data(f, 'easting_true_orig_pos', initial_pos, 4, '>f',
                                 (self.easting_true_orig))
                    _unpack_data(f, 'central_meridian_sf_pos', initial_pos, 4, '>f',
                                 (self.central_meridian_sf))
                    _unpack_data(f, 'experiment_no_pos', initial_pos, 2, '>h', (self.experiment_no))
                    _unpack_data(f, 'threshold_value_pos', initial_pos, 4, '>f', (self.thresh_value))
                    _unpack_data(f, 'threshold_value_alt_pos', initial_pos, 4, '>f',
                                 (self.thresh_value_alt))
                    _unpack_data(f, 'thresh_type_pos', initial_pos, 2, '>h', (self.thresh_type))
                    _unpack_data(f, 'prob_method_pos', initial_pos, 2, '>h', (self.prob_method))
                    _unpack_data(f, 'prob_niters_pos', initial_pos, 2, '>h', (self.prob_niters))
                    _unpack_data(f, 'prob_nmembers_pos', initial_pos, 2, '>h', (self.prob_nmembers))
                    _unpack_data(f, 'prob_threshwindow_pos', initial_pos, 2, '>h',
                                 (self.prob_threshwindow))
                    _unpack_data(f, 'prob_neighradius_pos', initial_pos, 4, '>f',
                                 (self.prob_neighradius))
                    _unpack_data(f, 'prob_radius_pos', initial_pos, 4, '>f', (self.prob_radius))
                    _unpack_data(f, 'prob_rfalpha_pos', initial_pos, 4, '>f', (self.prob_rfalpha))
                    _unpack_data(f, 'prob_fuzzyth_pos', initial_pos, 4, '>f', (self.prob_fuzzyth))
                    _unpack_data(f, 'prob_fuzzytime_pos', initial_pos, 4, '>f', (self.prob_fuzzytime))
                    _unpack_data(f, 'soil_type_pos', initial_pos, 2, '>h', (self.soil_type))
                    _unpack_data(f, 'ellipsoid', initial_pos, 2, '>h', (self.ellipsoid))
                    # set new inital_pos for next field in file
                    totfieldbytes = (self.colnumbers[cnt - 1] *
                                     self.rownumbers[cnt - 1] *
                                     datasize[cnt - 1])
                    initial_pos = initial_pos + paddedheaderbytes + totfieldbytes
                    if debug:
                        print("""Finished reading field {}. Next field starts at byte {}.
                              """.format(cnt, initial_pos))
                except Exception as E:
                    if ignoreErrors:
                        if not quiet:
                            print('ERROR: Could not read header number', cnt, ':', E)
                        warnings.warn('Not all headers loaded')
                        break
                    else:
                        print('ERROR: Could not read header number', cnt, ':', E)
                    raise

        f.close()

def _unpack_data(f, position, initial_pos, datalength, data_unpack_type, datalist):
    """A function used to read in header data from a position in the file
        and return the data storted there."""
    # Set positions in header for inforation that needs reading in:
    pos_dict = {'data_pos': 24,
                'byte_pos': 26,
                'row_pos': 32,
                'col_pos': 34,
                'field_pos': 38,
                'vert_coord_type_pos': 40,
                'vert_value_pos': 64,
                'vertupper_coord_type_pos': 42,
                'vertupper_value_pos': 68,
                'grid_pos': 30,
                'yorigin_pos': 72,
                'yresol_pos': 76,
                'xorigin_pos': 80,
                'xresol_pos': 84,
                'lat_true_orig_pos': 108,
                'long_true_orig_pos': 112,
                'easting_true_orig_pos': 116,
                'northing_true_orig_pos': 120,
                'central_meridian_sf_pos': 124,
                'rmdi_pos': 88,
                'unit_scalefactor_pos': 92,
                'unit_pos': 356,
                'chead_pos': 364,
                'imdi_pos': 50,
                'averaging_pos': 62,
                'aver_period_pos': 52,
                'aver_period_pos2': 508,
                'thresh_type_pos': 412,
                'prob_method_pos': 414,
                'prob_niters_pos': 416,
                'prob_nmembers_pos': 418,
                'prob_threshwindow_pos': 420,
                'prob_neighradius_pos': 224,
                'prob_radius_pos': 228,
                'prob_rfalpha_pos': 232,
                'prob_fuzzyth_pos': 236,
                'prob_fuzzytime_pos': 240,
                'soil_type_pos': 424,
                'experiment_no_pos': 28,
                'threshold_value_pos': 132,
                'threshold_value_alt_pos': 128,
                'ellipsoid': 56}
    seek_pos = initial_pos + pos_dict[position] + 2
    f.seek(seek_pos)
    temp = f.read(datalength)
    data = unpack(data_unpack_type, temp)[0]
    datalist.append(data)
    return datalist

if __name__ == "__main__":

    print(_get_averaging(8192+256+16+4+2))
    # test reading in data from diffent files into mycubes.
    filepath = "/data/users/frust/Nimrod_data_for_plotting/example_data/"
    # myfile = "201508270600_u1096_ng_umqv_temperature_2km"
    # myfile = "201508270600_e3195_u32_ume4_temperature_5km"
    myfile = "201510080000_globe_ll_umqg_temperature_25km"
    # myfile = "201508270300_u1096_ng_ek06_temperature_2km"

    print('\n Reading in Nimrod file into mycubes')
    mycubes = nimrod_to_cubes(filepath + myfile)
    # print(mycubes)

    # print('\n Select mycubes with field code 58, screen temperature')
    # cubelist1 = select_mycubes(mycubes, field_code=58)
    # print(cubelist1)

    # print('\n Select mycubes with field code 58, and cell method is None')
    # cubelist1 = select_mycubes(mycubes, field_code=58, cell_methods=['None'])
    # print(cubelist1)

    # Set up list of forecast lead time that will be plotted:
    max_fctime = int(5 * 60)
    max1hr = 2880     # Beyond this, 3-hrly images are drawn
    max3hr = 12000      # Beyond this 6-hrly images are draw
    fc_time_list = []
    if max_fctime < max1hr:
        fc_time_list = [z for z in range(0, max_fctime + 60, 60)]
    elif max_fctime >= max1hr and max_fctime <= max3hr:
        fc_time_list = [z for z in range(
            0, max1hr + 60, 60)] + [z for z in range(max1hr, max_fctime + 180, 180)]
    elif max_fctime > max3hr:
        fc_time_list = [z for z in range(0, max1hr + 60, 60)] + [z for z in range(
            max1hr, max3hr + 180, 180)] + [z for z in range(max3hr, max_fctime + 360, 360)]

    print('''\n Select mycubes with field code 58, cell method is None
             and limited set of forecast lead times''')
    cubelist1 = mycubes
    print(cubelist1)
    print(cubelist1[0])
    iris.save(cubelist1, './Test_cube.nc')
    mycubes = iris.load('./Test_cube.nc')
    print(mycubes)
    print(mycubes[0])