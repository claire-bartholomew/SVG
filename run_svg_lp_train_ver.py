#!/usr/bin/env python
# coding: utf-8

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
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pdb
import re
from datetime import datetime, timedelta
import time
import matplotlib.animation as manimation

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import gc

batch_size = 3 #, type=int, help='batch size')
data_root = 'data' #', help='root directory for data')
log_dir = 'logs' #, help='directory to save generations to')
seed = 1 #', default=1, type=int, help='manual seed')
n_past = 3 #', type=int, default=3, help='number of frames to condition on')
n_future = 21 #', type=int, default=7, help='number of frames to predict')
num_threads = 0 #', type=int, default=0, help='number of data loading threads')
nsample = 30 #', type=int, default=100, help='number of samples')
N = 256 #', type=int, default=256, help='number of samples')

n_eval = n_past+n_future
max_step = n_eval

random.seed(seed)
torch.manual_seed(seed)
dtype = torch.FloatTensor

#===============================================================================
def main(startdate, model_path, model, domain, threshold):

    print('Model = ', model_path, model)
    startdate = datetime.strptime('201807270000', '%Y%m%d%H%M') # for running inference for verification
    enddate = datetime.strptime('201901010000', '%Y%m%d%H%M') # for running inference for verification
    dtime = startdate

    frame_predictor, posterior, prior, encoder, decoder, last_frame_skip = load_model(model_path, model)

    while True:
        if dtime == enddate:
            break
        else:
            print(dtime)
            date_list = [dtime + timedelta(minutes=x*5) for x in range(36)]
            files_v = []
            for dt in date_list:
                dt_str = datetime.strftime(dt, '%Y%m%d%H%M')
                #files_v.append('/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str))
                files_v.append('/data/cr1/cbarth/phd/SVG/training_data/100days/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str))

            #rainy_dates = [#'0102', '0103', '0104', '0114', '0115', '0116', '0117', '0121',
            #   '0122', '0123', '0124', '0128', '0130', '0131', '0208', '0209',
            #   '0210', '0212', '0214', '0218', '0304', '0305', '0309', '0310',
            #   '0311', '0314', '0315', '0322', '0326', '0327', '0329', '0330',
            #   '0401', '0402', '0403', '0404', '0409', '0424', '0427', '0501',
            #   '0512', '0602', '0613', '0619', '0727', '0728', '0729', '0809',
            #   '0810', '0811', '0812', '0815', '0818', '0824', '0826', '0910',
            #   '0911', '0915', '0917', '0918', '0919', '0920', '0922', '1007',
            #   '1008', '1011', '1012', '1013', '1014', '1031', '1102', '1103',
            #   '1106', '1107', '1108', '1109', '1110', '1112', '1113', '1120',
            #   #'1127',
            #   '1128', '1129', '1130', '1201', '1202', '1204', '1205',
            #   '1206', '1207', '1208', '1215', '1216', '1217', '1218', '1219',
            #   '1220', '1221']
            #files_t = [f'/data/cr1/cbarth/phd/SVG/training_data/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
            #          for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates]

            list_tst = []
            for file in files_v:
                if os.path.isfile(file):
                    list_tst.append(file)

            if list_tst != []:
                test_loader, cube, start_date, skip = prep_data(list_tst, n_eval, domain, threshold)
                if skip == False:
                    testing_batch_generator = get_testing_batch(test_loader)

                    # Create cubes of right sizes (and scale for cbar by multiplying by 32)
                    #pred_cube = cube[:, 288:416, 100:228] #[:, 160:288, 130:258]
                    pred_cube = cube[:, domain[0]:domain[1], domain[2]:domain[3]]
                    pred_cube *= 32.

                    i = 0
                    print('start datetime:', start_date[i])
                    yyyy = str(start_date[i])[10:14]
                    mm = str(start_date[i])[15:17]
                    dd = str(start_date[i])[18:20]
                    hh = str(start_date[i])[21:23]
                    mi = str(start_date[i])[24:26]
                    dt_str = '{}{}{}{}{}'.format(yyyy, mm, dd, hh, mi)
                    # generate predictions
                    test_x = next(testing_batch_generator)
                    ssim, x, posterior_gen, all_gen = make_gifs(test_x, 'test', frame_predictor, posterior, prior, encoder, decoder, last_frame_skip)

                    batch_number = 0
                    # Find index of sample with highest SSIM score
                    mean_ssim = np.mean(ssim[batch_number], 1)
                    ordered = np.argsort(mean_ssim)
                    sidx = ordered[-1]
                    for t in range(n_eval):
                        pred_cube.data[t] = all_gen[sidx][t][batch_number][0].detach().numpy() * threshold
                        pred_cube.units = 'mm/hr'
                        print("plots_nn_T{}_{}.nc".format(dt_str, model[:-4]))
                    iris.save(pred_cube, "/data/cr1/cbarth/phd/SVG/model_output/{}_train/plots_nn_T{}_{}.nc".format(model[:-4], dt_str, model[:-4]))
            gc.collect()
            dtime = dtime + timedelta(minutes=15)

def load_model(model_path, model):
    # ---------------- load the models  ----------------
    tmp = torch.load('{}{}'.format(model_path, model), map_location='cpu')
    frame_predictor = tmp['frame_predictor']
    posterior = tmp['posterior']
    prior = tmp['prior']
    frame_predictor.eval()
    prior.eval()
    posterior.eval()
    encoder = tmp['encoder']
    decoder = tmp['decoder']
    encoder.train()
    decoder.train()
    frame_predictor.batch_size = batch_size
    posterior.batch_size = batch_size
    prior.batch_size = batch_size
    g_dim = tmp['opt'].g_dim
    z_dim = tmp['opt'].z_dim
    num_digits = tmp['opt'].num_digits

    # ---------------- set the options ----------------
    dataset = tmp['opt'].dataset
    last_frame_skip = tmp['opt'].last_frame_skip
    channels = tmp['opt'].channels
    image_width = tmp['opt'].image_width

    return frame_predictor, posterior, prior, encoder, decoder, last_frame_skip

# --------- load a dataset ------------------------------------
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def prep_data(files, n_eval, domain, threshold):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M"
    #regex = re.compile("^/data/cr1/cbarth/phd/SVG/verification_data/radar/(\d*)")
    regex = re.compile("^/data/cr1/cbarth/phd/SVG/training_data/100days/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files1 = sorted(files, key=gettimestamp)

    # only keep filenames where the right number of  consecutive files exist at 5 min intervals
    sorted_files = list(sorted_files1[0:0+n_eval]) #chunks(sorted_files1, n_eval))

    dataset = []
    fn = sorted_files
    cube = iris.load(fn)
    if len(cube) > 1:
        for i, cu in enumerate(cube):
            if np.shape(cu.coord('time'))[0] == 1:
                cube[i] = iris.util.new_axis(cu, 'time')
        cube = cube.merge()

    cube = cube[0] / 32. #Convert to mm/hr
    cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
    data = cube1.data

    if len(data) < n_eval:
        print('small data of size ', len(data))
        skip = True
        loader = []
        start_date = []
    else:
        skip = False
        #data = data[:, 160:288, 130:258] #focusing on a 128x128 grid box area over England
        data = data[:, domain[0]:domain[1], domain[2]:domain[3]]
        # Set limit of large values - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > threshold)] = threshold
        # Normalise data
        data = data / threshold
        start_date = cube.coord('forecast_reference_time')[0]
        dataset.append(data)
        dataset.append(data)
        dataset.append(data)
        # Convert to torch tensors
        tensor = torch.stack([torch.Tensor(i) for i in dataset])
        loader = DataLoader(tensor, #batch_size=1)
                            #num_workers=opt.data_threads,
                            batch_size=batch_size,
                            shuffle=False, #True, #False to keep same order of data
                            #drop_last=True,
                            pin_memory=True)

    return loader, cube1, start_date, skip

# -------------------------------------------------------------
def get_testing_batch(test_loader):
     while True:
         for i, sequence in enumerate(test_loader):
             if np.shape(sequence)[0] == batch_size:
                 batch = utils.normalize_data_gen(dtype, sequence)
                 yield batch

# --------- eval funtions ------------------------------------
def make_gifs(x, name, frame_predictor, posterior, prior, encoder, decoder, last_frame_skip):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()

        if last_frame_skip or i < n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            x_in = x[i]
            posterior_gen.append(x_in)
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

    ssim = np.zeros((batch_size, nsample, n_future))
    psnr = np.zeros((batch_size, nsample, n_future))
    all_gen = []
    for s in range(nsample):
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]

        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, n_eval):
            h = encoder(x_in)
            if last_frame_skip or i < n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]

                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()

                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    return(ssim, x, posterior_gen, all_gen)

if __name__ == "__main__":
    startdate = datetime.strptime('201909291200', '%Y%m%d%H%M')
    model_path = '/scratch/cbarth/phd/'
    model = 'model624800.pth' #131219.pth'
    domain = [160, 288, 130, 258]
    threshold = 64.
    main(startdate, model_path, model, domain, threshold)