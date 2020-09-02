import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
import iris
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import iris.plot as iplt
import numpy as np
import pdb
import os
import re
from datetime import datetime, timedelta
import gc

#===============================================================================
#def main():
def main(dtime, threshold): #model_path, model,

    #rainy_dates = ['0929'] #'0814', #'201908141630', '201909291300'] #1215', '1218', '1206'] #'1127','1109'] # '1108', '1109', '1110', '1112', '1113','1120', '1127', '1128', '1129', '1130','1202', '1204', '1205', '1206', '1207', '1208', '1215', '1216','1217', '1218', '1219', '1220', '1221', '1222']
    ## List all possible radar files in range and find those that exist #test or train dir
    #files_t = [f'/data/cr1/cbarth/phd/SVG/verification_data/radar/2019{mmdd}{h:02}{mi:02}_nimrod_ng_radar_rainrate_composite_1km_UK' \
    #           for mi in range(0,60,5) for h in range(13,24) for mmdd in rainy_dates]
    #list_test = []
    #for file in files_t:
    #     if os.path.isfile(file):
    #        list_test.append(file)
    #test_loader, cube = prep_data(list_test, 'test') #'train')
    startdate = dtime + timedelta(minutes=10)
    enddate = startdate + timedelta(minutes=15)

    n_eval = 13

    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('/scratch/cbarth/phd/milesial_unet_10ep_0.01lr_new.pt')) #milesial_unet_uk_16ep_0.01lr_lowres.pt')) ##milesial_unet_uk_15ep_0.01lr_h.pt'))
    model.eval()

    while True:
        if dtime == enddate:
            break
        else:
            print(dtime)
            date_list = [dtime + timedelta(minutes=x*5) for x in range(36)]
            files_v = []
            for dt in date_list:
                dt_str = datetime.strftime(dt, '%Y%m%d%H%M')
                files_v.append('/data/cr1/cbarth/phd/SVG/verification_data/radar/{}_nimrod_ng_radar_rainrate_composite_1km_UK'.format(dt_str))

            list_tst = []
            for file in files_v:
                if os.path.isfile(file):
                    list_tst.append(file)

            test_loader, cube, start_date, skip = prep_data(list_tst, n_eval, threshold)
            print(cube)
            cube.remove_coord('experiment_number')
            cube.remove_coord('realization')

            if skip == False:
                startdate = dtime + timedelta(minutes=10)
                dt_str = datetime.strftime(startdate,'%Y%m%d%H%M')
                print('start datetime of predictions:', dt_str)
                sequence = show_outputs(model, test_loader, n_eval)
                for i in range(n_eval):
                    j = i+2
                    cube.data[i] = sequence[0,j].detach().numpy() * 32
                    cube.units = 'mm/hr'
                print("unet_output_T{}.nc".format(dt_str))
                iris.save(cube, "/data/cr1/cbarth/phd/SVG/model_output/unet/unet_output_T{}.nc".format(dt_str))
            dtime = dtime + timedelta(minutes=15)

#===============================================================================
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
#===============================================================================
def prep_data(files, n_eval, threshold):
    batch_size = 20
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
        #data = data[:, domain[0]:domain[1], domain[2]:domain[3]]
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
        loader = utils.DataLoader(tensor, #batch_size=1)
                            #num_workers=opt.data_threads,
                            batch_size=batch_size,
                            shuffle=False, #True, #False to keep same order of data
                            #drop_last=True,
                            pin_memory=True)

    return loader, cube1, start_date, skip

#=============================================================================
def prep_data_original(files, folder):

    # Regrid to a resolution x4 lower
    sample_points = [('projection_y_coordinate', np.linspace(-624500., 1546500., 543)),
                     ('projection_x_coordinate', np.linspace(-404500., 1318500., 431))]

    timeformat = "%Y%m%d%H%M" # this is how your timestamp looks like
    #regex = re.compile("^/nobackup/sccsb/radar/(\d*)")

    if folder == 'train':
        regex = re.compile("^/data/cr1/cbarth/phd/SVG/verification_data/radar/(\d*)")
    elif folder == 'test':
        regex = re.compile("^/data/cr1/cbarth/phd/SVG/verification_data/radar/(\d*)")

    def gettimestamp(thestring):
        m = regex.search(thestring)
        return datetime.datetime.strptime(m.groups()[0], timeformat)

    # sort files by datetime
    sorted_files = sorted(files, key=gettimestamp)

    # only keep filenames where 16 consecutive files exist at 5 min intervals
    sorted_files = list(chunks(sorted_files, 16))
    for group in sorted_files:
        if len(group) < 16:
            sorted_files.remove(group)
        else:
            t0 = group[0].find('2019')
            dt1 = datetime.datetime.strptime(group[0][t0:t0+12], '%Y%m%d%H%M')
            t3 = group[15].find('2019')
            dt2 = datetime.datetime.strptime(group[15][t3:t3+12], '%Y%m%d%H%M')
            if (dt2-dt1 != datetime.timedelta(minutes=75)):
                print(dt2-dt1, 'remove files')
                sorted_files.remove(group)

    dataset = []
    for fn in sorted_files:
        print(fn)
        cube = iris.load_cube(fn)
        cube = cube / 32.
        cube1 = cube.interpolate(sample_points, iris.analysis.Linear())
        data = cube1.data

        # Set limit of large values # or to missing? - have asked Tim Darlington about these large values
        data[np.where(data < 0)] = 0.
        data[np.where(data > 32)] = 32.
        #data[np.where(data > 64)] = 64. #-1./32

        # Normalise data
        data = data / 32.
        #data = data / 64.

        # Binarise data
        #dataset[np.where(dataset < 0)] = 0.
        #dataset[np.where(dataset > 0)] = 1.

        dataset.append(data)
    #pdb.set_trace()
    # Convert to torch tensors
    tensor = torch.stack([torch.Tensor(i) for i in dataset])
    loader = utils.DataLoader(tensor, batch_size=1)

    return loader, cube1

#===============================================================================
# full assembly of the sub-parts to form the complete net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x #torch.sigmoid(x)

#===============================================================================
# sub-parts of the U-Net model
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        #print(x2.size(), x1.size())
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        #print('outconv finished')
        return x

#===============================================================================
def show_outputs(net, loader, n_eval): #, cube):
    count = 0
    #cube = cube[0] # Select just one timestep of cube
    for b, data in enumerate(loader):
        truth = data[:]
        data = data.type('torch.FloatTensor')
        # Wrap tensors in Variables
        inputs = Variable(data[:,:3])

        count += 1
        #Forward pass
        val_outputs = net(inputs) #* 64.
        val_outputs[np.where(val_outputs < 0.)] = 0.
        #val_outputs[np.where(val_outputs > 32.)] = 32.

        #add to sequence of radar images
        sequence = torch.cat((inputs, val_outputs), 1)

        for step in range(n_eval): #4): #12):
            gc.collect()
            print('step = {}'.format(step))

            sequence = sequence.type('torch.FloatTensor')
            inputs = sequence[:,-3:]

            #sequence = predict_1hr(sequence, net)
            val_outputs = predict_1hr(inputs, net)
            sequence = torch.cat((sequence, val_outputs), 1)

        return sequence

        # colors = ['black', 'cornflowerblue', 'royalblue', 'blue', 'lime', 'yellow', 'orange', 'red', 'fuchsia'] #, 'white']
        # levels = [0, 0.1, 0.25, 0.5, 1., 2., 4., 8. ,16., 32.]
        #
        # for i in range(8): #16):
        #     print('start figure', i)
        #     fig = plt.figure(figsize=(6, 8))
        #     # Add axes to the figure, to place the colour bar [left, bottom, width, height] (of cbar)
        #     colorbar_axes = fig.add_axes([0.15, 0.1, 0.73, 0.03])
        #     ax = fig.add_subplot(1,2,1)
        #     cube.data = truth[0,i].detach().numpy() * 32
        #     cf = iplt.contourf(cube, levels, colors=colors, origin='lower', extend='max')
        #     cf.cmap.set_over('white')
        #     plt.gca().coastlines('50m', color='white')
        #     ax = fig.add_subplot(1,2,2)
        #     cube.data = sequence[0,i].detach().numpy() * 32
        #     cf = iplt.contourf(cube, levels, colors=colors, origin='lower', extend='max')
        #     #cf = iplt.contourf(cube, cmap=plt.cm.Blues, vmin=0, vmax=1) #32)
        #     plt.gca().coastlines('50m', color='white')
        #     #plt.setp(ax.xaxis.get_ticklabels(), visible=False)
        #     #plt.setp(ax.yaxis.get_ticklabels(), visible=False)
        #     plt.title('U-Net T+{}'.format(i*5-10))
        #     plt.tight_layout()
        #     cbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal')
        #     cbar.ax.set_xlabel('Rain rate (mm/hr)')
        #     #plt.savefig('/home/home01/sccsb/radar_seq/img3/truth{}_im{}.png'.format(b, i))
        #     plt.savefig('/scratch/cbarth/phd/comparison_{}_im{}.png'.format(b, i))
        #     plt.close()

#===============================================================================
def predict_1hr(inputs, net): #sequence, net):
    #sequence = sequence.type('torch.FloatTensor')
    #inputs = sequence[:,-3:]

    #Wrap tensors in Variables
    inputs = Variable(inputs)
    #Forward pass
    val_outputs = net(inputs)
    val_outputs[np.where(val_outputs < 0.)] = 0.
    #sequence = torch.cat((sequence, val_outputs), 1)

    return val_outputs #sequence
#===============================================================================
if __name__ == "__main__":
    startdate = datetime.strptime('201910292350', '%Y%m%d%H%M')
    #model_path = '/scratch/cbarth/phd/'
    #model = 'model131219.pth'
    #domain = [160, 288, 130, 258]
    threshold = 32.
    main(startdate, threshold) # model_path, model,
