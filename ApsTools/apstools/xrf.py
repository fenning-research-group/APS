import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib.transforms as transforms

packageDir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(packageDir, 'include', 'xrfEmissionLines.json'), 'r') as f:
	emissionLines = json.load(f)

def AddXRFLines(elements, ax = None, maxlinesperelement = 4, tickloc = 'bottom', tickstagger = 0, ticklength = 0.05):        
    if ax is None:
    	ax = plt.gca()
    
    xlim0 = ax.get_xlim()
    stagger = 0
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for idx, element in enumerate(elements):
        color = plt.cm.get_cmap('tab10')(idx)
        ax.text(0.99, 0.01 + 0.05*idx, element,
            fontname = 'Verdana', 
            fontsize = 12,
            fontweight = 'bold',
            color = color, 
            transform = ax.transAxes,
            horizontalalignment = 'right',
            verticalalignment = 'bottom')

        plotted = 0
        for line in emissionLines[element]['xrfEmissionLines']:
            if (line <= xlim0[1]) and (line >= xlim0[0]):
#                 plt.plot([line, line], [0.98 - (idx+1)*ticklength, 0.98 - idx*ticklength], transform = trans, color = color, linewidth = 1.5)
                if tickloc == 'bottom':
                	plt.plot([line, line], [0.01 + stagger, 0.01 + ticklength + stagger], transform = trans, color = color, linewidth = 1.5)
                else:
                	plt.plot([line, line], [0.99 - stagger, 0.99 - ticklength - stagger], transform = trans, color = color, linewidth = 1.5)
            plotted += 1
            if plotted >= maxlinesperelement:
                break
        stagger += tickstagger