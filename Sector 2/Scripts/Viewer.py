#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

data_filepath = 'G:\\My Drive\\FRG\\Projects\\APS\\2IDD_2019\\Sample Data - 150C HEP'
data_files = os.listdir(data_filepath)


# In[ ]:



# In[2]:


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


# In[3]:


import pandas as pd

df_a =  pd.DataFrame(list(zip(x,y)), columns = ['X', 'Y'])
df_b = pd.DataFrame.from_records(xrf, columns = channels)
df_a['Scan'] = scan_nums
df_b['Scan'] = scan_nums

xrf = pd.merge(df_a, df_b, on = ['Scan'])


# In[4]:


xrf['X']


# In[5]:


# import altair as alt
# alt.renderers.enable('notebook')
# alt.data_transformers.enable('json')
# import numpy as np

# plotdat = xrf.loc[xrf['Scan'] == 67]
# x,y  = np.meshgrid(plotdat['X'][0], plotdat['Y'][0])

# source = pd.DataFrame({'x': x.ravel(),
#                        'y': y.ravel(),
#                        'z': plotdat['Si'][0].ravel()})

# alt.Chart(source).mark_rect().encode(
#     alt.X('x:O', bin = True),
#     alt.Y('y:O', bin = True),
#     alt.Color('z:Q')
# )


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent

class PickStack():
    def __init__(self, stack, on_pick):
        self.stack = stack
        self.ax = [artist.axes for artist in self.stack][0]
        self.on_pick = on_pick
        self.cid = self.ax.figure.canvas.mpl_connect('button_press_event', self.pick_action)
        self.ax.autoscale(enable = True)
        
    def pick_action(self, event):
        if not event.inaxes:
            return
        
        cont = [a for a in self.stack if a.contains(event)[0]]
        if not cont:
            return
        
        pick_event = PickEvent("pick_Event", self.ax.figure.canvas, event, cont[0], guiEvent = event.guiEvent, **cont[0].contains(event)[1])
        self.on_pick(pick_event)

        #stackoverflow.com/questions/56015753/picking-a-single-artist-from-a-set-of-overlapping-artists-in-matplotlib
    


# In[8]:


#get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from matplotlib.text import Text
from matplotlib import cm


## overview map
# fig, ax = plt.subplots(1,3)

fig = plt.figure(figsize = (10, 10), )
ax1 = plt.subplot2grid((2,2),(0,1), colspan = 2, rowspan = 1)
ax0 = plt.subplot2grid((2,2),(1,1), colspan = 2, rowspan = 1)
ax2 = plt.subplot2grid((2,4),(0,0), colspan = 1, rowspan = 4)
plt.tight_layout()

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

## picker functions
    
def plotxrf(ax, scan_number, channel):
    channeldat = xrf[xrf['Scan'] == int(scan_number)][channel]
    channeldat_np = np.array(channeldat)

    x = xrf[xrf['Scan'] == int(scan_number)]['X'].values
    y = xrf[xrf['Scan'] == int(scan_number)]['Y'].values

    ax.imshow(channeldat_np[0], extent =[x[0][0], x[0][-1], y[0][0], y[0][-1]])
    ax.set_title(channel + ', scan ' + str(scan_number))
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

## scan map

# fig_scan, ax_scan = plt.subplots()

ax1.set(xlabel = 'X Position (um)', ylabel = 'Y Position (um)')
ax0.set(xlabel = 'X Position (um)', ylabel = 'Y Position (um)')



# plotxrf(ax1, 64, 'Si')
plt.show()



plt.ion()
# In[ ]:


# from matplotlib import cm
# from collections import OrderedDict
# import numpy as np

# cmaps = OrderedDict()

# cm.get_cmap('Paired')(1)


# In[ ]:


# d['MAPS']['x_axis'][1]


# In[ ]:




