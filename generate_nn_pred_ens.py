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
matplotlib.use('tkAgg') #Agg') #'TkAgg')
import matplotlib.pyplot as plt

import sys; sys.path.append('/home/h03/jcheung/python/lib')
import toolbox as tb

#===============================================================================
def main(startdate): #, enddate, mod, n_past, n_future, ts=5):

    mod = 'model624800' #1755653' #model1734435'
    ts = 5
    n_past = 3 #7', type=int, default=3, help='number of frames to condition on')
    n_future = 21 #17 #21 #', type=int, default=7, help='number of frames to predict')
    batch_size = 3 #, type=int, help='batch size')
    data_root = 'data' #', help='root directory for data')
    #model_path = 'logs/lp/radar/model=dcgan128x128-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=True-beta=0.0001000/model4.pth'
    #model_path = 'logs/lp/radar/model=vgg128x128-rnn_size=256-predictor-posterior-prior-rnn_layers=2-1-1-n_past=3-n_future=7-lr=0.0020-g_dim=128-z_dim=10-last_frame_skip=False-beta=0.0001000/model3.pth'
    model_path = '/scratch/cbarth/phd/{}.pth'.format(mod) #model1755653.pth' #model1734435.pth' #530043.pth' #131219.pth' #582525.pth' #need to change line 79 too
    log_dir = 'logs' #, help='directory to save generations to')
    seed = 1 #', default=1, type=int, help='manual seed')
    num_threads = 0 #', type=int, default=0, help='number of data loading threads')
    nsample = 30 #', type=int, default=100, help='number of samples')
    N = 256 #', type=int, default=256, help='number of samples')

    n_eval = n_past+n_future
    max_step = n_eval

    #print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(opt.seed)
    dtype = torch.FloatTensor

    # ---------------- load the models  ----------------
    tmp = torch.load(model_path, map_location='cpu')
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

    #mod = 'model1755653' #model1734435' #need to change line 33 too
    #startdate = datetime.strptime('201909290000', '%Y%m%d%H%M')
    enddate = datetime.strptime('201910010000', '%Y%m%d%H%M')
    dtime = startdate

    #while True:

    dt_check = dtime + timedelta(minutes=n_past*ts) # start cube at final input frame
    date_check = datetime.strftime(dt_check, '%Y%m%d%H%M')
    date_list = [dtime + timedelta(minutes=x*5) for x in range(n_eval)] #36)]
    #date_list = [dtime + timedelta(minutes=x*30) for x in range(36)]
    files_v = []
    for dt in date_list:
        dt_str = datetime.strftime(dt, '%Y%m%d%H%M')
        files_v.append('/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str))

    list_tst = []
    for file in files_v:
        if os.path.isfile(file):
            list_tst.append(file)

    test_loader, cube, start_date, skip = prep_data(list_tst, n_eval, batch_size)
    #print('start datetime:', start_date[0])
    yyyy = str(start_date[0])[10:14]
    mm = str(start_date[0])[15:17]
    dd = str(start_date[0])[18:20]
    hh = str(start_date[0])[21:23]
    mi = str(start_date[0])[24:26]
    dt_str0 = '{}{}{}{}{}'.format(yyyy, mm, dd, hh, mi)
    date0 = datetime.strptime(dt_str0, '%Y%m%d%H%M')
    date = date0 + timedelta(minutes=n_past*ts) # start cube at final input frame
    dt_str = datetime.strftime(date, '%Y%m%d%H%M')

    if skip == False:
        print(dtime, date_check)
        testing_batch_generator = get_testing_batch(test_loader, batch_size, dtype)

        # Create cubes of right sizes (and scale for cbar by multiplying by 32)
        pred_cube = cube[:, 160:288, 130:258]
        pred_cube *= 32.
        pred_cube = pred_cube[n_past-1:n_eval]

        # generate predictions
        test_x = next(testing_batch_generator)
        ssim, x, posterior_gen, all_gen = make_gifs(test_x, 'test',
                 frame_predictor, posterior, n_eval, encoder, decoder,
                 last_frame_skip, n_past, n_future, nsample, batch_size,
                 prior)

        batch_number = 0 #in range(1): #batch_size):
        # Find index of sample with highest SSIM score
        #mean_ssim = np.mean(ssim[0], 1)
        #mean_ssim = np.mean(ssim[batch_number], 1)
        #ordered = np.argsort(mean_ssim)
        #sidx = ordered[-1]
        #rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for ensemble in range(30):
            for t in range(n_past-1, n_eval): # just save predictions
                pred_cube.data[t-n_past+1] = all_gen[ensemble][t][batch_number][0].detach().numpy() * 64.
                #print('mean = ', np.mean(pred_cube.data))
                pred_cube.units = 'mm/hr'
            #print("/data/cr1/cbarth/phd/SVG/model_output/{}_ens2/plots_nn_T{}_{}_ens{}.nc".format(mod, dt_str, mod, ensemble))
            iris.save(pred_cube, "/data/cr1/cbarth/phd/SVG/model_output/{}_ens3/plots_nn_T{}_{}_ens{}.nc".format(mod, dt_str, mod, ensemble))

    dtime = dtime + timedelta(minutes=15)

# --------- load a dataset ------------------------------------
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def prep_data(files, n_eval, batch_size):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M"
    regex = re.compile("^/data/cr1/cbarth/phd/SVG/verification_data/radar/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files1 = sorted(files, key=gettimestamp)

    # only keep filenames where the right number of  consecutive files exist at 5 min intervals
    sorted_files = list(sorted_files1[0:0+n_eval]) #chunks(sorted_files1, n_eval))

    dataset = []
    #for fn in sorted_files:
    fn = sorted_files #[0]
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
        start_date = cube.coord('forecast_reference_time')[0]
    elif np.mean(data) > 0.01: #0.5: #0.1: # limit loading to rainy days to speed up extraction
        skip = False
        data = data[:, 160:288, 130:258] #focusing on a 128x128 grid box area over England
        # Set limit of large values - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > 64)] = 64.
        # Normalise data
        data = data / 64.
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
    else:
        skip = True
        loader = []
        start_date = cube.coord('forecast_reference_time')[0]

    return loader, cube1, start_date, skip

# -------------------------------------------------------------
def get_testing_batch(test_loader, batch_size, dtype):
     while True:
         for i, sequence in enumerate(test_loader): #.dataset:
             #print(np.shape(sequence))
             if np.shape(sequence)[0] == batch_size:
                 batch = utils.normalize_data_gen(dtype, sequence)
                 yield batch

# --------- eval funtions ------------------------------------
def make_gifs(x, name, frame_predictor, posterior, n_eval, encoder, decoder,
              last_frame_skip, n_past, n_future, nsample, batch_size, prior):
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
            #posterior_gen.append(x[i])
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
    startdates = []
    startdate = datetime.strptime('201912290005', '%Y%m%d%H%M')   #05  #09
    mod = 'model624800'
    #startdate = datetime.strptime('202008271205', '%Y%m%d%H%M')
    for d in range (0, 360): #1): #335): #30): #, 3):
        for m in range(96):
            startdates.append(startdate + timedelta(days = d) + timedelta(minutes = m*15))

    for date in startdates:
        print(date)
        dt_str = datetime.strftime(date, '%Y%m%d%H%M')
        if not os.path.exists('/data/cr1/cbarth/phd/SVG/model_output/{}_ens3/plots_nn_T{}_{}_ens1nc'.format(mod, dt_str, mod)):
            main(date)
