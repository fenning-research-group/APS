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


## build plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from matplotlib.text import Text
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors

## picker functions
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plotxrf(ax, scan_number, channel):
    channeldat = xrf[xrf['Scan'] == int(scan_number)][channel]
    channeldat_np = np.array(channeldat)

    x = xrf[xrf['Scan'] == int(scan_number)]['X'].values
    y = xrf[xrf['Scan'] == int(scan_number)]['Y'].values

    ax.clear()
    color = cm.get_cmap('viridis')
    color_trimmed = truncate_colormap(color, minval = 0.0, maxval = 0.99)

    im = ax.imshow(channeldat_np[0], 
        extent =[x[0][0], x[0][-1], y[0][0], y[0][-1]],
        cmap = color_trimmed,
        interpolation = 'none')

    scalebar = ScaleBar(1e-6,
        color = [1, 1, 1],
        box_color = [1, 1, 1],
        box_alpha = 0,
        location = 'lower right',
        border_pad = 0.05)

    ax.text(0.02, 0.98, str(scan_number) + ': ' + channel,
        fontname = 'Verdana', 
        fontsize = 10,
        color = [1, 1, 1], 
        transform = ax.transAxes,
        horizontalalignment = 'left',
        verticalalignment = 'top')

    ax.text(0.98, 0.98, str(int(np.amax(channeldat_np[0]))) + '\n' + str(int(np.amin(channeldat_np[0]))),
        fontname = 'Verdana', 
        fontsize = 10,
        color = [1, 1, 1], 
        transform = ax.transAxes,
        horizontalalignment = 'right',
        verticalalignment = 'top')    

    ax.set_title(channel + ', scan ' + str(scan_number))
    ax.add_artist(scalebar)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    return
    
def onpick(event):
    if isinstance(event.artist, Rectangle):
        for each in r:
            each.set_alpha(0.1)
            
        patch = event.artist
        patch.set_alpha(0.5)
        global plotted_scan
        plotted_scan = patch.get_label()
        plotxrf(ax1, plotted_scan, channel)
        plt.draw()

def oncheckbox(label):
    global channel, plotted_scan
    channel = label
    if plotted_scan:
        plotxrf(ax1, plotted_scan, channel)

fig = plt.figure(figsize = (10, 6), )
ax1 = plt.subplot2grid((2,2),(0,1), colspan = 2, rowspan = 1)
ax0 = plt.subplot2grid((2,2),(1,1), colspan = 2, rowspan = 1)
ax2 = plt.subplot2grid((2,4),(0,0), colspan = 1, rowspan = 4)
plt.tight_layout()

# build overview map rectangles
r = []
box_params = []

for x,y,scan in xrf[['X', 'Y', 'Scan']].itertuples(index = False):
    corner = (min(x), min(y))
    x_width = max(x) - min(x)
    y_width = max(y) - min(y)
    scan_area = (x_width * y_width)
    box_params.append([scan_area, corner, x_width, y_width, scan])

box_params.sort(reverse = True)
    
for each_box in box_params:
    color = cm.get_cmap('Set1')(len(r))
    hr = Rectangle(each_box[1], each_box[2], each_box[3], 
                    picker = True, 
                    facecolor = color, 
                    alpha = 0.2, 
                    edgecolor = [0, 0, 0],
                    label = each_box[4])
    r.append(ax0.add_patch(hr))

#check boxes

rax = ax2
global channel, plotted_scan
plotted_scan = []
labels = list(xrf)[3:]
channel = labels[0]
visibility = np.zeros((len(labels),1))
check  = CheckButtons(rax, labels, visibility)
check.on_clicked(oncheckbox)

ax0.autoscale(enable = True)
cid = fig.canvas.mpl_connect('pick_event', onpick)


ax1.set(xlabel = 'X Position (um)', ylabel = 'Y Position (um)')
ax0.set(xlabel = 'X Position (um)', ylabel = 'Y Position (um)')
plt.show()
plt.ion()


