## find files
import os

data_filepath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP'
data_files = os.listdir(data_filepath)

## read files
import h5py

def read_2idd_h5(f):
    with h5py.File(f, 'r') as dat:
        xvals = dat['MAPS']['x_axis'][:]
        yvals = dat['MAPS']['y_axis'][:]    
        xrf = dat['MAPS']['XRF_roi'][:]

    return xvals, yvals, xrf
    
f = os.path.join(data_filepath, data_files[5])

with h5py.File(f, 'r') as data:
    channels = data['MAPS']['channel_names'][:].astype('U13')
    energy = data['MAPS']['energy'][:]

x = []
y = []
xrf = []
scan_nums = []

for filename in data_files:
    if '2idd_' in filename:
        f = os.path.join(data_filepath, filename)
        x_data, y_data, xrf_data = read_2idd_h5(f)
        scan_num = int(filename[5:9])
        
        scan_nums.append(scan_num)
        x.append(x_data)
        y.append(y_data)
        xrf.append(xrf_data)


##generat dataframe from files
import pandas as pd

df_a =  pd.DataFrame(list(zip(x,y)), columns = ['X', 'Y'])
df_b = pd.DataFrame.from_records(xrf, columns = channels)
df_a['Scan'] = scan_nums
df_b['Scan'] = scan_nums

xrf = pd.merge(df_a, df_b, on = ['Scan'])


## build plots
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from matplotlib.text import Text
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import json

## picker functions
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plotxrf(scan_number, channel):
    channeldat = xrf[xrf['Scan'] == int(scan_number)][channel]
    channeldat_np = np.array(channeldat)

    x = xrf[xrf['Scan'] == int(scan_number)]['X'].values
    y = xrf[xrf['Scan'] == int(scan_number)]['Y'].values

    color = cm.get_cmap('viridis')
    color_trimmed = self.truncate_colormap(color, minval = 0.0, maxval = 0.99)

    fig = plt.figure(figsize = (2, 2))
    ax = plt.gca()

    im = ax.imshow(channeldat_np[0][2:-2][2:-2], 
        extent =[x[0][0], x[0][-1], y[0][0], y[0][-1]],
        cmap = color_trimmed,
        interpolation = 'none')

    ## text + scalebar objects
    opacity = 1

    scalebar = ScaleBar(1e-6,
        color = [1, 1, 1, opacity],
        box_color = [1, 1, 1],
        box_alpha = 0,
        location = 'lower right',
        border_pad = 0.1)

    ax.text(0.02, 0.98, str(scan_number) + ': ' + channel,
        fontname = 'Verdana', 
        fontsize = 12,
        color = [1, 1, 1, opacity], 
        transform = ax.transAxes,
        horizontalalignment = 'left',
        verticalalignment = 'top')

    ax.text(0.98, 0.98, str(int(np.amax(channeldat_np[0]))) + '\n' + str(int(np.amin(channeldat_np[0]))),
        fontname = 'Verdana', 
        fontsize = 12,
        color = [1, 1, 1, opacity], 
        transform = ax.transAxes,
        horizontalalignment = 'right',
        verticalalignment = 'top')    

    # ax.set_title(channel + ', scan ' + str(scan_number))
    ax.add_artist(scalebar)
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    return fig

def plotoverview(scan_number, box_params):
    fig = plt.figure(figsize = (3, 3))
    ax = plt.gca()
    color_counter = 0
    for each_box in box_params:

        if each_box[4] == scan_number:
            opacity = 0.8
        else:
            opacity = 0.15

        color = cm.get_cmap('Set1')(color_counter)
        hr = Rectangle(each_box[1], each_box[2], each_box[3], 
                        picker = True, 
                        facecolor = color, 
                        alpha = opacity, 
                        edgecolor = [0, 0, 0],
                        label = each_box[4])
        ax.add_patch(hr)
        color_counter = color_counter + 1
    ax.autoscale(enable = True)
    plt.tight_layout()
    return fig

# def getscanparameters(xrf):
#     box_params = xrf[['X', 'Y', 'Scan']]

#     for x,y,scan in xrf[['X', 'Y', 'Scan']].itertuples(index = False):
#         corner = (min(x), min(y))
#         x_width = max(x) - min(x)
#         y_width = max(y) - min(y)
#         stepsize = x_width / len(x)
#         scan_area = (x_width * y_width)
#         box_params.append([scan_area, corner, x_width, y_width, stepsize, scan])

#     return box_params


export_filepath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP\\export'

if not os.path.exists(export_filepath): 
    os.mkdir(export_filepath)

# scan_params = getscanparameters(xrf)

box_params = []
for x,y,scan in xrf[['X', 'Y', 'Scan']].itertuples(index = False):
    corner = (min(x), min(y))
    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    scan_area = (x_width * y_width)
    box_params.append([scan_area, corner, x_width, y_width, scan])

box_params.sort(reverse = True)
    

# generate + export plots to output directory
for scan in scan_nums:
    if not os.path.exists(os.path.join(export_filepath, str(scan))):
        os.mkdir(os.path.join(export_filepath, str(scan)))

    print(scan)

    # export scan parameters
    x = xrf[xrf['Scan'] == scan]['X'].values[0]
    y = xrf[xrf['Scan'] == scan]['Y'].values[0]

    params = {  'x_range': int(max(x) - min(x)),
                'y_range': int(max(y) - min(y)),
                'stepsize': str((max(x)-min(x))/(len(x)-1))
                }

    writestr = json.dumps(params)
    fname = os.path.join(export_filepath, str(scan), 'scanparameters.json')
    with open(fname, "w") as f:
        f.write(writestr)
        f.close() 

    # export xrf images
    for channel in channels:
        fig = plotxrf(scan, channel)
        f = os.path.join(export_filepath, str(scan), 'images')
        if not os.path.exists(f):
            os.mkdir(f)
        writepath = os.path.join(f, channel)
        plt.savefig(writepath + '.jpeg', format='jpeg', dpi=300)
        plt.close()

    # export overview image

    fname = os.path.join(export_filepath, str(scan), 'overview')
    fig = plotoverview(scan, box_params)
    plt.savefig(fname + '.jpeg', format = 'jpeg', dpi = 300)
