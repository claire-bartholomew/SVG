import iris
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import pdb
import numpy as np
import re
import datetime
import time

# --------- load a dataset ------------------------------------

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def prep_data(files, filedir):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M"
    regex = re.compile("^training_data/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files = sorted(files, key=gettimestamp)

    # only keep filenames where 10 consecutive files exist at 5 min intervals
    sorted_files = list(chunks(sorted_files, 10))
    for group in sorted_files:
        if len(group) < 10:
            sorted_files.remove(group)
        else:
            t0 = group[0].find('2018')
            dt1 = datetime.datetime.strptime(group[0][t0:t0+12], '%Y%m%d%H%M')
            t9 = group[9].find('2018')
            dt2 = datetime.datetime.strptime(group[9][t9:t9+12], '%Y%m%d%H%M')
            if (dt2-dt1 != datetime.timedelta(minutes=45)):
                print(dt2-dt1, 'remove files')
                sorted_files.remove(group)
    smalldata_count = 0
    dataset = []
    count = 0

    for fn in sorted_files:
        print(fn)
        #cube = iris.load_cube(fn)
        cube = iris.load(fn)
        cube = cube[0] / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data
        #pdb.set_trace()
        data = data[:, 160:288, 130:258] #focusing on a 128x128 grid box area over England

        ## Set limit of large values - have asked Tim Darlington about these large values
        #data[np.where(data < 0)] = 0.
        #data[np.where(data > 32)] = 32.

        ## Normalise data
        #data = data / 32.

        if len(data) < 10:
            print(fn)
            print('small data of size ', len(data))
            smalldata_count += 1
        else:
            dataset.append(data)
            count += 1

    print('small data count', smalldata_count)

    print('size of data:', len(dataset), np.shape(dataset), count)

    np.save('test.npy', dataset)

    mean = np.mean(dataset)
    std = np.std(dataset)
    mini = np.amin(dataset)
    maxi = np.amax(dataset)
    print('Dataset statistics')
    print('max = ', maxi)
    print('min = ', mini)
    print('mean = ', mean)
    print('std = ', std)

    pdb.set_trace()

rainy_dates = ['0102', '0103', '0104', '0114', '0115', '0116', '0117', '0121'] #,
               #'0122', '0123', '0124', '0128', '0130', '0131', '0208', '0209',
               #'0210', '0212', '0214', '0218', '0304', '0305', '0309', '0310',
               #'0311', '0314', '0315', '0322', '0326', '0327', '0329', '0330',
               #'0401', '0402', '0403', '0404', '0409', '0424', '0427', '0501',
               #'0512', '0602', '0613', '0619', '0727', '0728', '0729', '0809',
               #'0810', '0811', '0812', '0815', '0818', '0824', '0826', '0910',
               #'0911', '0915', '0917', '0918', '0919', '0920', '0922', '1007',
               #'1008', '1011', '1012', '1013', '1014', '1031', '1102', '1103',
               #'1106', '1107', '1108', '1109', '1110', '1112', '1113', '1120',
               #'1127', '1128', '1129', '1130', '1201', '1202', '1204', '1205',
               #'1206', '1207', '1208', '1215', '1216', '1217', '1218', '1219',
               #'1220', '1221']

#val_dates = ['1222']

# List all possible radar files in range and find those that exist
files_t = [f'training_data/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
           for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates] #d in range(25) for mo in range(5,6)]

list_train = []
for file in files_t:
    if os.path.isfile(file):
        list_train.append(file)
train_loader = prep_data(list_train, 'train')
print('stats generated')
