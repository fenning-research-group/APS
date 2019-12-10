import matplotlib.pyplot as plt
import numpy as np
import json
import os
import matplotlib.transforms as transforms

packageDir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(packageDir, 'include', 'xrfEmissionLines.json'), 'r') as f:
	emissionLines = json.load(f)

def AddXRFLines(elements, ax = None, maxlinesperelement = 7, tickloc = 'bottom'):        
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

        for line in emissionLines[element]['xrfEmissionLines']:
            if (line <= maxlinesperelement) and (line >= 1):
#                 plt.plot([line, line], [0.98 - (idx+1)*step, 0.98 - idx*step], transform = trans, color = color, linewidth = 1.5)
                if tickloc == 'bottom':
                	plt.plot([line, line], [0.01, 0.01 + step], transform = trans, color = color, linewidth = 1.5)
                else:
                	plt.plot([line, line], [0.99, 0.99 - step], transform = trans, color = color, linewidth = 1.5)