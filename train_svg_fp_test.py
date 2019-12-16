import iris
import matplotlib.pyplot as plt
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
import progressbar
import numpy as np
import re
import datetime
import time
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/fp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')


opt = parser.parse_args()
if opt.model_dir != '':
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-rnn_layers=%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%d-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
#torch.cuda.manual_seed_all(opt.seed)
dtype = torch.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)

import models.lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
        
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu, logvar):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= opt.batch_size  
  return KLD


# --------- transfer to gpu ------------------------------------
#frame_predictor.cuda()
#posterior.cuda()
#encoder.cuda()
#decoder.cuda()
#mse_criterion.cuda()

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# --------- load a dataset ------------------------------------
def prep_data(files, filedir):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M"
    if filedir == 'train':
        regex = re.compile("^/nobackup/sccsb/radar/train/(\d*)")
    elif filedir == 'test':
        regex = re.compile("^/nobackup/sccsb/radar/test/(\d*)")
    elif filedir == 'may':
        regex = re.compile("^/nobackup/sccsb/radar/may/(\d*)")

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
        print(fn)
        cube = iris.load_cube(fn)
        cube = cube / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data
        data = data[:, :128, :128]
        #print(np.shape(data))

        # Set limit of large values - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > 32)] = 32. 

        # Normalise data
        data = data / 32.

        dataset.append(data)
 
    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])

    ## Add 1 dimension
    #tensor = tensor.unsqueeze(0)

    loader = DataLoader(tensor) #, #batch_size=1)
                        #num_workers=opt.data_threads,
                        #batch_size=opt.batch_size,
                        #shuffle=True,
                        #drop_last=True,
                        #pin_memory=True)

    #import pdb; pdb.set_trace()

    return loader

#files_t = [f'/nobackup/sccsb/radar/may/2018{mo:02}{d:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
#               for mi in range(0,60,5) for h in range(24) for d in range(25) for mo in range(5,6)]
#list_train = []
#for file in files_t:
#    if os.path.isfile(file):
#        list_train.append(file)

#train_loader = prep_data(list_train, filedir='may') #train')
#
#files_v = [f'/nobackup/sccsb/radar/may/2018{mo:02}{d:02}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
#           for mi in range(0,60,5) for h in range(24) for d in range(25,28) for mo in range(5,6)]
#list_val = []
#for file in files_v:
#    if os.path.isfile(file):
#        list_val.append(file)
#test_loader = prep_data(list_val, filedir='may') #test')

rainy_dates = ['0102', '0103', '0104', '0114', '0115', '0116', '0117', '0121',
               '0122', '0123', '0124', '0128', '0130', '0131', '0208', '0209',
               '0210', '0212', '0214', '0218', '0401', '0402', '0403', '0404',
               '0409', '0424', '0427', '0501', '0512', '0727', '0728', '0729',
               '0810', '0811', '0812', '0815', '0818', '0824', '0826',
               '1007', '1008', '1011', '1012', '1013', '1014', '1031', '1102',
               '1103', '1106', '1107'] #, '0809', '1108', '1109'] #, '1110', '1112', '1113'
                #'1120', '1127', '1128', '1129', '1130']
val_dates = ['0304', '0305', '0309'] #, '0310', '0311', '0314', '0315', '0322',
                 #'0326', '0327', '0329', '0330', '0602', '0613', '0619', '0910',
                 #'0911', '0915', '0917', '0918', '0919', '0920', '0922', '1201']#,
                 ##'1202', '1204', '1205', '1206', '1207', '1208', '1215', '1216',
                 ##'1217', '1218', '1219', '1220', '1221', '1222']

# List all possible radar files in range and find those that exist
files_t = [f'/nobackup/sccsb/radar/train/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
           for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates] #d in range(25) for mo in range(5,6)]

list_train = []
for file in files_t:
    if os.path.isfile(file):
        list_train.append(file)
train_loader = prep_data(list_train, 'train')

#import pdb; pdb.set_trace()

files_v = [f'/nobackup/sccsb/radar/test/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
           for mi in range(0,60,5) for h in range(24) for mmdd in val_dates] #range(25,28) for mo in range(5,6)]
list_val = []
for file in files_v:
    if os.path.isfile(file):
        list_val.append(file)
val_loader = prep_data(list_val, 'test')


#train_data, test_data = utils.load_dataset(opt)
#train_loader = DataLoader(train_data,
#                          num_workers=opt.data_threads,
#                          batch_size=opt.batch_size,
#                          shuffle=True,
#                          drop_last=True,
#                          pin_memory=True)
#test_loader = DataLoader(test_data,
#                         num_workers=opt.data_threads,
#                         batch_size=opt.batch_size,
#                         shuffle=True,
#                         drop_last=True,
#                         pin_memory=True)

def get_training_batch():
    print('getting training batch')
    while True:
        #print('true')
        #batch = utils.normalize_data(opt, dtype, train_loader.dataset)
        #yield batch
        for sequence in train_loader.dataset:
            #print(np.shape(sequence))
            batch = utils.normalize_data(opt, dtype, sequence)
            #print(np.shape(batch))
            #pdb.set_trace()
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader.dataset:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 5 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    h_seq = [encoder(x[i]) for i in range(opt.n_past)]
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h_seq[i-1]
                h = h.detach()
            elif i < opt.n_past:
                h, _ = h_seq[i-1]
                h = h.detach()
            if i < opt.n_past:
                z_t, _, _ = posterior(h_seq[i][0])
                frame_predictor(torch.cat([h, z_t], 1)) 
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t = torch.cuda.FloatTensor(opt.batch_size, opt.z_dim).normal_()
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        for s in range(nsample):
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for s in range(nsample):
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    utils.save_gif(fname, gifs)


def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    h_seq = [encoder(x[i]) for i in range(opt.n_past+opt.n_future)]
    for i in range(1, opt.n_past+opt.n_future):
        h_target = h_seq[i][0].detach()
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
        h = h.detach()
        z_t, mu, logvar = posterior(h_target)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
   
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()


    #=========================
    # delete after
    print(np.shape(x[0]))
    print(np.shape(x[1]))
    print(opt.n_past+opt.n_future)

    for i in range(opt.n_past+opt.n_future):
        print(i)
        h_tst = encoder(x[i])
    #=======================


    #x = x[0]
    h_seq = [encoder(x[i]) for i in range(opt.n_past+opt.n_future)]
    mse = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        print(i)
        h_target = h_seq[i][0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h_seq[i-1]
        else:
            h = h_seq[i-1][0]
        z_t, mu, logvar = posterior(h_target)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar)

    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    #progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        #progress.update(i+1)
        #print(i)
        x = next(training_batch_generator)

        #print(np.shape(x[0]))
        #pdb.set_trace()

        # train frame_predictor 
        mse, kld = train(x)
        epoch_mse += mse
        epoch_kld += kld


    #progress.finish()
    #utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    x = next(testing_batch_generator)
    plot(x, epoch)
    plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

