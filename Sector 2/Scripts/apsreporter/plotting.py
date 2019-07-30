import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
import matplotlib.transforms as transforms
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import os
import numpy as np
import json

ROOT_DIR = os.path.dirname(__file__)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def plotxrf(outputfolder, scan, channel, x, y, xrf, overwrite = True, logscale = False):

	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, channel + '.' + image_format)

	if overwrite or not os.path.exists(savepath):
		## setup colormap + caxis scale
		color = cm.get_cmap('inferno')
		# color_trimmed = truncate_colormap(cmap = color, minval = 0.0, maxval = 0.99)	#exclude highest brightness pixels to improve contrast with overlay text
		color_trimmed = color

		excludedborder = 0.2
		m,n = xrf.shape
		mlim = int(np.rint(m * excludedborder))
		nlim = int(np.rint(n * excludedborder))
		vmin = np.amin(xrf[mlim:-mlim, nlim:-nlim])
		vmax = np.amax(xrf[mlim:-mlim, nlim:-nlim])

		fig = plt.figure(figsize = (2, 2))
		ax = plt.gca()

		if logscale:
			norm = colors.LogNorm()
		else:
			norm = None
							
		im = ax.imshow(xrf, 
			extent =[x[0], x[-1], y[0], y[-1]],
			cmap = color_trimmed,
			interpolation = 'none',
			vmin = vmin,
			vmax = vmax,
			norm = colors.LogNorm(),
			origin = 'lower')

		## text + scalebar objects
		opacity = 1

		scalebar = ScaleBar(1e-6,
			color = [1, 1, 1, opacity],
			box_color = [1, 1, 1],
			box_alpha = 0,
			location = 'lower right',
			border_pad = 0.1)

		ax.text(0.02, 0.98, str(scan) + ': ' + channel,
			fontname = 'Verdana', 
			fontsize = 12,
			color = [1, 1, 1, opacity], 
			transform = ax.transAxes,
			horizontalalignment = 'left',
			verticalalignment = 'top')

		ax.text(0.98, 0.98, str(round(np.amax(xrf),2)) + '\n' + str(round(np.amin(xrf),2)),
			fontname = 'Verdana', 
			fontsize = 12,
			color = [1, 1, 1, opacity], 
			transform = ax.transAxes,
			horizontalalignment = 'right',
			verticalalignment = 'top')    

		ax.add_artist(scalebar)
		plt.axis('equal')
		plt.axis('off')
		plt.gca().set_position([0, 0, 1, 1])
		plt.savefig(savepath, format=image_format, dpi=300)
		plt.close() 

	return savepath

def plotoverview(outputfolder, scan, scandat):
	#generate box bounds
	box_params = []
	for sc, val in scandat.items():
		x = val['x']
		y = val['y']

		corner = (min(x), min(y))
		x_width = max(x) - min(x)
		y_width = max(y) - min(y)
		scan_area = (x_width * y_width)

		box_params.append([scan_area, corner, x_width, y_width, sc])

	box_params.sort(reverse = True)


	# generate and save plot
	fig = plt.figure(figsize = (3, 3))
	ax = plt.gca()
	color_counter = 0
	for each_box in box_params:
		if each_box[4] == scan:
			opacity = 0.8	#highlight the selected scan
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
	plt.axis('equal')
	ax.title.set_text('Map Location Overview (um)')

	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, 'overviewmap.' + image_format)
	plt.savefig(savepath, format=image_format, dpi=300, bbox_inches = 'tight')
	plt.close() 

	return savepath

def plotintegratedxrf(outputfolder, scan, scandat):
	xrf = scandat['integratedxrf']
	low = 1	#low energy plotting cutoff, keV
	high = float(scandat['energy']) + 0.5 #high energy plotting cutoff, keV

	# generate and save plot
	plt.figure(figsize = (8,3.2))
	plt.plot(
		xrf[(xrf[:,0]>=low) & (xrf[:,0]<=high),0],
		xrf[(xrf[:,0]>=low) & (xrf[:,0]<=high),1],
		color = 'k',
		linewidth = 1)
	ax = plt.gca()
	plt.yscale('log')
	plt.xlabel('Energy (keV)')
	plt.ylabel('Counts (log)')
	ax.autoscale(enable = True, tight = True)
	plt.gca().xaxis.set_major_locator(MultipleLocator(1))
	plt.gca().xaxis.set_minor_locator(MultipleLocator(0.2))
	plt.grid(True)		
	ax.title.set_text('Integrated XRF Spectrum')

	# add tick marks for elements
	with open(os.path.join(ROOT_DIR, 'xrflines.json'), 'r') as f:
		emissionlines = json.load(f)

	elements = []
	for channel in list(scandat['xrf'].keys()):
		if (':' not in channel) and (channel is not 'XBIC'):	#skip the ratio maps and XBIC channel
			elements.append(channel.split('_')[0])	#exclude the emission line if present (ie I_L -> I)

	trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
	step = 0.05

	for idx, element in enumerate(elements):
		color = cm.get_cmap('tab10')(idx)
		ax.text(0.01, 0.98 - 0.08*idx, element,
			fontname = 'Verdana', 
			fontsize = 12,
			fontweight = 'bold',
			color = color, 
			transform = ax.transAxes,
			horizontalalignment = 'left',
			verticalalignment = 'top')

		for line in emissionlines[element]['xrfEmissionLines']:
			if (line <= high) and (line >= low):
				plt.plot([line, line], [1 - (idx+1)*step, 1 - idx*step], transform = trans, color = color, linewidth = 1.5)


	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, 'integrated.' + image_format)
	
	plt.savefig(savepath, format=image_format, dpi=300, bbox_inches = 'tight')
	plt.close() 

	return savepath

def plotcorrmat(outputfolder, scan, scandat):
	#build correlation matrix
	# flatdata = [item for sublist in scandat['xrf'][channels[0]] for item in sublist]	#flatten list of lists to 1d list
	# # flatdata = scandat['xrf'][channels[0]].flatten()
	# for key in channels[1:]:
	# 	newflatdata = [item for sublist in scandat['xrf'][key] for item in sublist]		#flatten list of lists to 1d list

	# 	flatdata = np.vstack((flatdata, newflatdata))
	channels = list(scandat['xrf'].keys())

	flatdata = scandat['xrf'][channels[0]].flatten()

	for key in channels[1:]:
		flatdata = np.vstack((flatdata, scandat['xrf'][key].flatten()))

	corrmat = np.corrcoef(flatdata)
	# generate and save plot
	fig, ax = plt.subplots(figsize = (3,3))
	im = ax.matshow(corrmat, cmap = cm.get_cmap('RdBu'), vmin = -1, vmax = 1)
	ax.set_xticks(np.arange(len(channels)))
	ax.set_yticks(np.arange(len(channels)))
	ax.set_xticklabels(channels, rotation = 45)
	ax.set_yticklabels(channels)
	ax.tick_params(axis = 'both', bottom = True, top = False, labelbottom = True, labeltop = False, which = 'major', labelsize = 10)
	ax.title.set_text('Correlation Matrix')

	cbar = fig.colorbar(im,
		ax = ax,
		shrink = 0.8,
		aspect = 30)
	cbar.set_label('Correlation Coefficient')
	cbar.ax.tick_params(labelsize = 8)

	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, 'correlationmatrix.' + image_format)
	
	plt.savefig(savepath, format=image_format, dpi=300, bbox_inches = 'tight')
	plt.close() 
	return savepath