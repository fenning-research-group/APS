import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import os
import numpy as np

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	new_cmap = colors.LinearSegmentedColormap.from_list(
		'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
		cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def plotxrf(outputfolder, scan, channel, x, y, xrf):
	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, channel + '.' + image_format)

	if not os.path.exists(savepath):
		xrf = np.array(xrf)
		xrf = xrf[:, :-2]	#remove last two lines, dead from 2idd flyscan

		color = cm.get_cmap('viridis')
		color_trimmed = truncate_colormap(cmap = color, minval = 0.0, maxval = 0.99)	#exclude highest brightness pixels to improve contrast with overlay text

		fig = plt.figure(figsize = (2, 2))
		ax = plt.gca()
		im = ax.imshow(xrf, 
			extent =[x[0], x[-3], y[0], y[-1]],
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

		ax.text(0.02, 0.98, str(scan) + ': ' + channel,
			fontname = 'Verdana', 
			fontsize = 12,
			color = [1, 1, 1, opacity], 
			transform = ax.transAxes,
			horizontalalignment = 'left',
			verticalalignment = 'top')

		ax.text(0.98, 0.98, str(int(np.amax(xrf))) + '\n' + str(int(np.amin(xrf))),
			fontname = 'Verdana', 
			fontsize = 12,
			color = [1, 1, 1, opacity], 
			transform = ax.transAxes,
			horizontalalignment = 'right',
			verticalalignment = 'top')    

		ax.add_artist(scalebar)
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
	
	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, 'overviewmap.' + image_format)
	plt.savefig(savepath, format=image_format, dpi=300)
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

	# color_counter = 0
	# for each_box in box_params:
	# 	if each_box[4] == scan:
	# 		opacity = 0.8	#highlight the selected scan
	# 	else:
	# 		opacity = 0.15

	# 	color = cm.get_cmap('Set1')(color_counter)
	# 	hr = Rectangle(each_box[1], each_box[2], each_box[3], 
	# 					picker = True, 
	# 					facecolor = color, 
	# 					alpha = opacity, 
	# 					edgecolor = [0, 0, 0],
	# 					label = each_box[4])
	# 	ax.add_patch(hr)
	# 	color_counter = color_counter + 1
	# ax.autoscale(enable = True)
	# plt.tight_layout()
	
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
	channels = list(scandat['xrf'].keys())
	flatdata = [item for sublist in scandat['xrf'][channels[0]] for item in sublist]	#flatten list of lists to 1d list
	# flatdata = scandat['xrf'][channels[0]].flatten()
	for key in channels[1:]:
		newflatdata = [item for sublist in scandat['xrf'][key] for item in sublist]		#flatten list of lists to 1d list

		flatdata = np.vstack((flatdata, newflatdata))
	
	corrmat = np.corrcoef(flatdata)
	
	# generate and save plot
	fig, ax = plt.subplots()
	im = ax.matshow(corrmat, cmap = cm.get_cmap('RdBu'))
	ax.set_xticks(np.arange(len(channels)))
	ax.set_yticks(np.arange(len(channels)))
	ax.set_xticklabels(channels)
	ax.set_yticklabels(channels)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
	
	savefolder = os.path.join(outputfolder, str(scan))
	if not os.path.exists(savefolder):
		os.mkdir(savefolder)

	image_format = 'jpeg'
	savepath = os.path.join(savefolder, 'correlationmatrix.' + image_format)
	
	plt.savefig(savepath, format=image_format, dpi=300, bbox_inches = 'tight')
	plt.close() 

	return savepath