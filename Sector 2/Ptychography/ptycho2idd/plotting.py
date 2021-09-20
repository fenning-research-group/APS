from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm


def draw(folder, scan, xmin, xmax, ymin, ymax, detector_width = 256, show = True, exportpath = None):
	files = os.listdir(folder)

	filepiece = [(x.split('_scan')[1],os.path.join(folder, x)) for x in files if '_scan' in x]
	scannum = [(x[0].split('_')[0],x[1]) for x in filepiece]
	goodfiles = [x[1] for x in scannum if int(x[0]) is scan]

	for file in goodfiles:
		if '_abs' in file:
			if '_probes' in file:
				probemodulus = np.array(Image.open(file))
			else:
				objmodulus = np.array(Image.open(file))
		elif '_ph' in file:
			if '_probes' in file:
				probephase = np.array(Image.open(file))
			else:
				objphase = np.array(Image.open(file))


	fig,ax = plt.subplots(1,2, figsize = (7,3))

	fig.suptitle('Scan ' + str(scan))
	d = int(detector_width/2)

	scalebar = ScaleBar(1e-6,
		color = [1, 1, 1, 0.5],
		box_color = [1, 1, 1],
		box_alpha = 0,
		location = 'lower right',
		border_pad = 0.1)

	im0 = ax[0].imshow(objmodulus[d:-d, d:-d],
		extent = [xmin, xmax, ymin, ymax],
		cmap = cm.get_cmap('cividis'),
		origin = 'lower'
	)
	ax[0].title.set_text('Amplitude')
	ax[0].add_artist(scalebar)
	ax[0].axis('off')

	scalebar = ScaleBar(1e-6,
		color = [1, 1, 1, 0.5],
		box_color = [1, 1, 1],
		box_alpha = 0,
		location = 'lower right',
		border_pad = 0.1)

	im1 = ax[1].imshow(objphase[d:-d, d:-d], 
		extent = [xmin, xmax, ymin, ymax],
		cmap = cm.get_cmap('cividis'),
		origin = 'lower'
	)
	ax[1].title.set_text('Phase')
	ax[1].add_artist(scalebar)
	cbar = fig.colorbar(im1,
		ax = ax[1],
		shrink = 1,
		aspect = 30)
	cbar.set_label('Phase (mrad)')
	cbar.ax.tick_params(labelsize = 8)
	ax[1].axis('off')
	
	plt.tight_layout()

	# ax[2].imshow(probemodulus)

	if show:
		plt.show()
 
	if exportpath:
		plt.savefig(exportpath, format='jpeg', dpi=300)
		plt.close()
