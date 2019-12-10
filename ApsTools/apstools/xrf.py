import matplotlib.pyplot as plt
import numpy as np
import json
import os

with open(os.path.join('include', 'xrfEmissionLines.json'), 'r') as f:
	emissionLines = json.load(f)

def AddXRFLines(elements, ax = None, maxlinesperelement = 7):        
    if ax is None:
    	ax = plt.gca()
    
    step = 0.05
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for idx, element in enumerate(elements):
        color = plt.cm.get_cmap('tab10')(idx)
        ax.text(0.99, 0.01 + step*idx, element,
            fontname = 'Verdana', 
            fontsize = 12,
            fontweight = 'bold',
            color = color, 
            transform = ax.transAxes,
            horizontalalignment = 'right',
            verticalalignment = 'bottom')
        
        
        for line in emissionlines[element]['xrfEmissionLines']:
            if (line <= maxlinesperelement) and (line >= 1):
#                 plt.plot([line, line], [0.98 - (idx+1)*step, 0.98 - idx*step], transform = trans, color = color, linewidth = 1.5)
                plt.plot([line, line], [0.01, 0.01 + step], transform = trans, color = color, linewidth = 1.5)