import iris
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
import itertools
import pdb
#import progressbar
import numpy as np
import re
import datetime
import time
import h5py 
from pathlib import Path
from torch.utils import data

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='radar', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=3, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=7, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=7, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', default=True, help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame') #action='store_true'


opt = parser.parse_args()
print(opt.last_frame_skip)

if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model7.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
#random.seed(opt.seed)
np.random.seed(opt.seed) #random seed testing - put this line in place of one above
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

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
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

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
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

print('------------models set up-----------')

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

print('-----------loss functions defined-----------')

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

# --------- load a dataset ------------------------------------
class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive=False, load_data=False, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            #for gname, group in h5_file.items():
                #for dname, ds in group.items():
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    idx = self._add_to_cache(ds.value, file_path)
                    
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'data' or 'label' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def prep_data(files): #, filedir):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M"
    #if filedir == 'train':
    regex = re.compile("^/nobackup/sccsb/radar/train/(\d*)")
    #elif filedir == 'test':
    #    regex = re.compile("^/nobackup/sccsb/radar/test/(\d*)")
    #elif filedir == 'may':
    #    regex = re.compile("^/nobackup/sccsb/radar/may/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files = sorted(files, key=gettimestamp)
    ##reduce to every 6th files, i.e. every half hour
    #sorted_files = sorted_files[0::6]

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
            if (dt2-dt1 != datetime.timedelta(minutes=45)): #45 for sequences of 5, 270 for sequences of 30
                print(dt2-dt1, 'remove files')
                sorted_files.remove(group)
    count = 0
    dataset = []
    for fn in sorted_files:
        #print(fn)
        #cube = iris.load_cube(fn)
        cube = iris.load(fn)
        cube = cube[0] / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data
        data = data[:, 160:288, 130:258] #focusing on a 128x128 grid box area over England
        #data = data[:, 288:416, 100:228] # scottish domain

        # limit range of data
        #data[np.where(data < 4)] = 0. #mask data to concentrate on higher rain rates
        data[np.where(data < 0)] = 0.
        data[np.where(data > 64)] = 64.

        #Log transform of data
        #data = np.log(data+1)

        # Normalise data
        data = data / 64.
        #data = data / np.log(64.)

        if len(data) < 10:
            print(fn)
            print('small data of size ', len(data))
            count += 1
        else:
            dataset.append(data)
    print(dataset)
    print(dataset[0])
    print('count', count)
    #print('data max', np.amax(dataset))
    #dataset = dataset / np.amax(dataset)

    print('size of data:', len(dataset), np.shape(dataset))

    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('dataset', data=dataset)
    h5f.close()

    #h5f2 = h5py.File('data.h5','r')
    #b = h5f2['dataset'][:]
    #h5f2.close()

    ## Convert to torch tensors
    #tensor = torch.stack([torch.Tensor(i) for i in dataset])

    transform_train = transforms.Compose([
        transforms.RandomRotation(180), #RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(rgb_mean, rgb_std),
    ])

    transformed_data = HDF5Dataset('/nobackup/sccsb/radar/h5', recursive=False, load_data=False,
                                   data_cache_size=4, transform=transform_train)

    #data_loader = data.DataLoader(dataset, **loader_params)

    loader = DataLoader(transformed_data, #tensor, #batch_size=1)
                        #num_workers=opt.data_threads,
                        batch_size=opt.batch_size,
                        shuffle=False, #True,
                        #train=True,
                        #transform=transform_train,
                        #drop_last=True,
                        pin_memory=True)

    return loader

rainy_dates = ['0102', '0103', '0104', '0114', '0115', '0116', '0117', '0121'] #,
#               '0122', '0123', '0124', '0128', '0130', '0131', '0208', '0209',
#               '0210', '0212', '0214', '0218', '0304', '0305', '0309', '0310',
#               '0311', '0314', '0315', '0322', '0326', '0327', '0329', '0330',
#               '0401', '0402', '0403', '0404', '0409', '0424', '0427', '0501',
#               '0512', '0602', '0613', '0619', '0727', '0728', '0729', '0809',
#               '0810', '0811', '0812', '0815', '0818', '0824', '0826', '0910',
#               '0911', '0915', '0917', '0918', '0919', '0920', '0922', '1007',
#               '1008', '1011', '1012', '1013', '1014', '1031', '1102', '1103',
#               '1106', '1107', '1108', '1109', '1110', '1112', '1113', '1120',
#               '1127', '1128', '1129', '1130', '1201', '1202', '1204', '1205',
#               '1206', '1207', '1208', '1215', '1216', '1217', '1218', '1219',
#               '1220', '1221']

#val_dates = ['1222']

# List all possible radar files in range and find those that exist
files_t = [f'/nobackup/sccsb/radar/train/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
           for mi in range(0,60,5) for h in range(24) for mmdd in rainy_dates] #d in range(25) for mo in range(5,6)]

list_train = []
for file in files_t:
    if os.path.isfile(file):
        list_train.append(file)
train_loader = prep_data(list_train) #, 'train')
print('training data loaded')

#files_v = [f'/nobackup/sccsb/radar/test/2018{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
#           for mi in range(0,60,5) for h in range(24) for mmdd in val_dates] #range(25,28) for mo in range(5,6)]
#list_tst = []
#for file in files_v:
#    if os.path.isfile(file):
#        list_tst.append(file)
#test_loader = prep_data(list_tst, 'test')

def get_training_batch():
    while True:
        for sequence in train_loader: #.dataset:  #train_loader
            if np.shape(sequence)[0] == opt.batch_size:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
training_batch_generator = get_training_batch()

#pdb.set_trace()

#def get_testing_batch():
#    while True:
#        for sequence in test_loader: #.dataset:
#            if np.shape(sequence)[0] == opt.batch_size:
#                batch = utils.normalize_data(opt, dtype, sequence)
#                yield batch 
#testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    nsample = 20 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
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
            print(t)
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx, 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
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
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        z_t, _, _= posterior(h_target)
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
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
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()

    mse = 0
    kld = 0
    for i in range(1, opt.n_past+opt.n_future):
        #print(i)
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])
        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

# --------- training loop ------------------------------------
print('Start training loop')
mse_loss = []
kld_loss = []
for epoch in range(opt.niter):
    print('epoch = {}'.format(epoch))
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    #progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        #print(i)
        #progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse, kld = train(x)
        epoch_mse += mse
        epoch_kld += kld

    #progress.finish()
    #utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    mse_loss.append(epoch_mse/opt.epoch_size)
    kld_loss.append(epoch_kld/opt.epoch_size)

    # plot some stuff
    frame_predictor.eval()
    #encoder.eval()
    #decoder.eval()
    posterior.eval()
    prior.eval()

    #x = next(testing_batch_generator)
    #plot(x, epoch)
    #plot_rec(x, epoch)

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model7.pth' % opt.log_dir)
    print('updated model saved')
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)

    print('MSE losses: ', mse_loss)
    print('KLD losses: ', kld_loss)
