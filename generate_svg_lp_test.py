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
#import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pdb
import re
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=30, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=3, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=7, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')


opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
#torch.cuda.manual_seed_all(opt.seed)
dtype = torch.FloatTensor



# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path, map_location='cpu')
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
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
#frame_predictor()
#posterior()
#prior()
#encoder()
#decoder()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


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
    regex = re.compile("^/data/cr1/cbarth/phd/SVG/verification_data/radar/(\d*)")
    #if filedir == 'train':
    #    regex = re.compile("^/nobackup/sccsb/radar/train/(\d*)")
    #elif filedir == 'test':
    #    regex = re.compile("^/nobackup/sccsb/radar/test/(\d*)")
    #elif filedir == 'may':
    #    regex = re.compile("^/nobackup/sccsb/radar/may/(\d*)")

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

    dataset = []
    for fn in sorted_files:
        cube = iris.load_cube(fn)
        cube = cube / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data
        data = data[:, 160:288, 130:258] #focusing on a 128x128 grid box area over England

        # Set limit of large values - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > 32)] = 32.

        # Normalise data
        data = data / 32.

        dataset.append(data)

    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])

    loader = DataLoader(tensor, #batch_size=1)
                        #num_workers=opt.data_threads,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        #drop_last=True,
                        pin_memory=True)
    return loader

#rainy_dates = ['0102', '0103', '0104', '0114', '0115', '0116', '0117', '0121',
#               '0122', '0123', '0124', '0128', '0130', '0131', '0208', '0209',
#               '0210', '0212', '0214', '0218', '0401', '0402', '0403', '0404',
#               '0409', '0424', '0427', '0501', '0512', '0727', '0728', '0729'] #,
#               #'0810', '0811', '0812', '0815', '0818', '0824', '0826',
#               #'1007', '1008', '1011', '1012', '1013', '1014', '1031', '1102',
#               #'1103', '1106', '1107'] #, '0809', '1108', '1109'] #, '1110', '1112', '1113'
#                #'1120', '1127', '1128', '1129', '1130']

#val_dates = ['0304'] #, '0305', '0309'] #, '0310', '0311', '0314', '0315', '0322',
                 #'0326', '0327', '0329', '0330', '0602', '0613', '0619', '0910',
                 #'0911', '0915', '0917', '0918', '0919', '0920', '0922', '1201']#,
                 ##'1202', '1204', '1205', '1206', '1207', '1208', '1215', '1216',
                 ##'1217', '1218', '1219', '1220', '1221', '1222']

# List all possible radar files in range and find those that exist
#files_t = [f'/nobackup/sccsb/radar/train/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
#           for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates] #d in range(25) for mo in range(5,6)]

#list_train = []
#for file in files_t:
#    if os.path.isfile(file):
#        list_train.append(file)
#train_loader = prep_data(list_train, 'train')
#print('training data loaded')

#files_v = [f'/nobackup/sccsb/radar/test/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
#           for mi in range(0,60,5) for h in range(24) for mmdd in val_dates]

files_v = [f'/data/cr1/cbarth/phd/SVG/verification_data/radar/2019{mm:02}{dd:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
           for mi in range(0,60,5) for h in range(24) for dd in range(32) for mm in range(8, 11)]

list_tst = []
for file in files_v:
    if os.path.isfile(file):
        list_tst.append(file)
test_loader = prep_data(list_tst, 'test')

#def get_training_batch():
#    while True:
#        for sequence in train_loader: #.dataset:  #train_loader
#            if np.shape(sequence)[0] == opt.batch_size:
#                batch = utils.normalize_data(opt, dtype, sequence)
#                yield batch
#training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader: #.dataset:
            if np.shape(sequence)[0] == opt.batch_size:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------

def make_gifs(x, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_eval):
        #import pdb; pdb.set_trace()
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()

        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)

        #pdb.set_trace()

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    #progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(nsample):
        #progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
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

    #progress.finish()
    #utils.clear_progressbar()

    ###### ssim ######
    for i in range(opt.batch_size):
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)
        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i)
        utils.save_gif_with_text(fname, gifs, text)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

for i in range(0, opt.N, opt.batch_size):
    ## plot train
    #train_x = next(training_batch_generator)
    ##pdb.set_trace()
    #make_gifs(train_x, i, 'train')

    # plot test
    test_x = next(testing_batch_generator)
    make_gifs(test_x, i, 'test')
    print(i)
