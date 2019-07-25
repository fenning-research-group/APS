from datetime import datetime
from tkinter import Tk, IntVar, Checkbutton, Button, mainloop, W
import h5py
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as colors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, Frame, FrameBreak, PageTemplate, NextPageTemplate, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as pilImage
import csv

class tableofcontents:
	## Class used to construct table of contents for report. Includes sections, subsection, and possible subsections, the lowest level of which contains scan numbers and channels to be put in report

	def __init__(self, title, description = None, outputfile = None, datafolder = None):
		self.title = title
		#self.title = title if title is not None else None
		self.description = description or None
		self.outputfile = outputfile or None
		self.datafolder = datafolder or None
		self._contents = {}

		# if datafolder is not None:
		# 	self.all_scans = self.get_all_scans()

	@property
	def contents(self):
		return self._contents
	@contents.setter
	def contents(self, newcontent):
		name, content = newcontent
		self._contents[name] = content
		# if name in self.contents:
		# 	self._contents[name] = content
		# else:
		# 	self._contents.update(name = content)	

	@property
	def energy(self):
		return self._energy
	@contents.setter
	def energy(self, energycsv):
		with open(energycsv, 'r') as csv_file:
			c = csv.reader(csv_file, delimiter = ',')
			line_counts = 0
			self._energy = {}
			for scan,row in enumerate(c):
				self._energy[scan+1] = row[0]

	@property
	def dwell(self):
		return self._dwell
	@contents.setter
	def dwell(self, dwellcsv):
		with open(dwellcsv, 'r') as csv_file:
			c = csv.reader(csv_file, delimiter = ',')
			line_counts = 0
			self._dwell = {}
			scancounter = 1
			for row in c:
				self._dwell[scancounter] = row[0]
				scancounter = scancounter + 1
	

	def show(self):

		def printwithtabs(printstr, tabs):
			tabstr = ''
			for each in range(tabs):
				tabstr = tabstr + '\t'
			print(tabstr + printstr)

		def showscanlist(scanlist, tabs):
			for scan, description in scanlist.scans.items():
				if description == None:
					description = ''
				printwithtabs('Scan {0:d}: {1:s}'.format(scan, description), tabs + 1)
		
		def showcomparison(comparison, tabs):
			printwithtabs('{0:s}'.format(comparison.description) ,tabs + 1)
			for scan, description in comparison._scans.items():
				if description == None:
					description = ''
				printwithtabs('Scan {0:d}: {1:s}'.format(scan, description), tabs + 2)

		def showsection(sections, tabs):
			for sec, cont in sections.items():
				# if type(content) is 'section':
				if isinstance(cont, section):
					printwithtabs('Section: {0:s}'.format(sec), tabs + 1)
					showsection(cont.contents, tabs + 1)
				# elif type(content) is 'scanlist':
				if isinstance(cont, scanlist):
					printwithtabs('Scans: {0:s}'.format(sec), tabs + 1)
					showscanlist(cont, tabs + 1)
				if isinstance(cont, comparison):
					printwithtabs('Comparison: {0:s}'.format(sec), tabs + 1)
					showcomparison(cont, tabs + 1)


		print('Title: {0:s}'.format(self.title))
		print('Description: {0:s}'.format(self.description))

		showsection(self._contents, tabs = 0)

	def export(self):

		def parsescanlist(scanlist):
			jscan = {}
			channels = scanlist._channels
			for scan, description in scanlist.scans.items():
				if description == None:
					description = ''
				jscan[scan] = {	
					'description': description,
					'channels': channels,
					'energy': self._energy[scan],
					'dwell': self._dwell[scan]
				}
			return jscan

		def parsecomparison(comparison):
			jcomparison = {
				'title': comparison.title,
				'description': comparison.description,
				'channels': comparison._channels
			}
			jscan = {}
			for scan, description in comparison._scans.items():
				if description == None:
					description = ''
				jscan[scan] = {
					'description': description,
					'energy': self._energy[scan],
					'dwell': self._dwell[scan]
				}
			jcomparison['scans'] = jscan

			return jcomparison

		def parsesection(sections):
			jsection = {}
			for sec, cont in sections.items():
				# if type(content) is 'section':
				if isinstance(cont, section):
					jsection[cont.title] = {
						'description': cont.description,
						'content': parsesection(cont.contents)
						}
				# elif type(content) is 'scanlist':
				if isinstance(cont, scanlist):
					jsection['scans'] = parsescanlist(cont)
				if isinstance(cont, comparison):
					jsection['comparison'] = parsecomparison(cont)

			return jsection

		j = {}

		j['Title'] = self.title
		j['Description'] = self.description
		j['DataFolder'] = self.datafolder
		j['Contents'] = parsesection(self._contents)
		writestr = json.dumps(j)

		with open(self.outputfile, 'w') as f:
			f.write(writestr)
			f.close()

		return self.outputfile
class section:

	def __init__(self, title, description = None):
		self.title = title
		self.description = description or None
		self._contents = {}

	@property
	def contents(self):
		return self._contents
	@contents.setter
	def contents(self, newcontent):
		name, content = newcontent
		self._contents[name] = content

		# if name in self._contents:
		# 	self._contents[name] = content
		# else:
		# 	self._contents[name] = content
class scanlist:

		def __init__(self, datafolder, scans):
			self.datafolder = datafolder
			if type(scans) is dict:
				self._scans = scans
			else:
				self._scans = {scan:None for scan in scans}
			# self.descriptions = [None] * len(_scans)


		@property
		def scans(self):
			return self._scans
		@scans.setter
		def scans(self, scan_nums):
			self._scans = scan_nums

		@property
		def channels(self):
			return self._channels
		# @channels.setter
		def setchannels(self, channels = None):
			if channels:
				self._channels = channels
			elif len(self._scans) == 0:
				raise ValueError('No scans assigned to this scan list: set first using .scans = [scan list here]')
			else:
				def pick_channels(channels):
					# tkinter gui to select channels
					master = Tk()				
					max_rows = 5
					var = []
					for i, channel in enumerate(channels):
						colnum = int(np.max([0,np.floor((i-1)/max_rows)]))
						rownum = int(np.mod(i, max_rows))
						var.append(IntVar())
						Checkbutton(master, text=channel, variable=var[i]).grid(row=rownum, column = colnum, sticky=W)
					Button(master, text = 'Select Channels', command = master.quit()).grid(row = 6, sticky = W)
					mainloop()

					selected_channels = [channels[x] for x in range(len(channels)) if var[x].get() == 1]

					return selected_channels

				scanfids = os.listdir(self.datafolder)
				scanfids = [x for x in scanfids if '2idd_' in x]
				scan_nums = [int(x[5:9]) for x in scanfids]		#5:9 are the positions for scan number in filename string. Can improve this later, or modify for different file format in the future

				scans = {x:y for x,y in zip(scan_nums,scanfids)}

				first_scan = list(self._scans.keys())[0]

				f = os.path.join(self.datafolder, scans[first_scan])	#open first scan h5 file to check which channels are available
				with h5py.File(f, 'r') as data:
					all_channels = data['MAPS']['channel_names'][:].astype('U13').tolist()

				all_channels.append('XBIC')	#this option links to the downstream ion chamber scaler, typically used for recording XBIC current
				self._channels = pick_channels(all_channels)

		def description(self, scan, description):
			if scan in self._scans:
				self._scans[scan] = description
			else:
				print('Scan %d not in scan list.'.format(scan))

		# @property
		# def descriptions(self):
		# 	return self.descriptions
		# @descriptions.setter
		# def descriptions(self, scan, description):
		# 	scan_index = self.scans.index(scan)
		# 	self.descriptions[scan_index] = description
class comparison:

		def __init__(self, datafolder, scans, title = None, description = None):
			self.datafolder = datafolder
			self.title = title or None
			self.description = description or None
			if type(scans) is dict:
				self._scans = scans
			else:
				self._scans = {scan:None for scan in scans}
				# self._comparisons = {comparison:filler for comparison in comparisons}
			# self.descriptions = [None] * len(_scans)


		@property
		def scans(self):
			return self._scans
		@scans.setter
		def scans(self, scan_nums):
			self._scans = scan_nums

		@property
		def channels(self):
			return self._channels
		# @channels.setter
		def setchannels(self):
			print(self.title)
			if channels:
				self._channels = channels
			elif len(self._scans) == 0:
				raise ValueError('No scans assigned to this comparison list')
			else:
				def pick_channels(channels):
					# tkinter gui to select channels
					master = Tk()				
					max_rows = 5
					var = []
					for i, channel in enumerate(channels):
						colnum = int(np.max([0,np.floor((i-1)/max_rows)]))
						rownum = int(np.mod(i, max_rows))
						var.append(IntVar())
						Checkbutton(master, text=channel, variable=var[i]).grid(row=rownum, column = colnum, sticky=W)
					Button(master, text = 'Select Channels', command = master.quit()).grid(row = 6, sticky = W)
					mainloop()

					selected_channels = [channels[x] for x in range(len(channels)) if var[x].get() == 1]

					return selected_channels

				scanfids = os.listdir(self.datafolder)
				scanfids = [x for x in scanfids if '2idd_' in x]
				scan_nums = [int(x[5:9]) for x in scanfids]		#5:9 are the positions for scan number in filename string. Can improve this later, or modify for different file format in the future

				scans = {x:y for x,y in zip(scan_nums,scanfids)}

				first_scan = list(self._scans.keys())[0]

				f = os.path.join(self.datafolder, scans[first_scan])	#open first scan h5 file to check which channels are available
				with h5py.File(f, 'r') as data:
					all_channels = data['MAPS']['channel_names'][:].astype('U13')

				all_channels.append('XBIC')	#this option links to the downstream ion chamber scaler, typically used for recording XBIC current
				self._channels = pick_channels(all_channels)

		# def description(self, scan, description):
		# 	if scan in self._scans:
		# 		self._scans[scan] = description
		# 	else:
		# 		print('Scan %d not in scan list.'.format(scan))
class build:

	def __init__(self, tableofcontents, outputfolder, title, boundaries = False):
		with open(tableofcontents, 'r') as f:
			self.tableofcontents = json.load(f)
		self.outputfolder = outputfolder
		f = os.path.join(outputfolder, title)
		self.doc = SimpleDocTemplate(f,
			showBoundary = boundaries)
		self.Story = []
		self.title = title

	def readscans(self, scanlist):		
		def read_2idd_h5(f):
			with h5py.File(f, 'r') as dat:
				xvals = dat['MAPS']['x_axis'][:]
				yvals = dat['MAPS']['y_axis'][:]    
				xrf = dat['MAPS']['XRF_roi'][:].tolist()
				energy = dat['MAPS']['energy']
				int_spec = dat['MAPS']['int_spec']
				summed_xrf = np.column_stack((energy, int_spec))

				scaler_names = dat['MAPS']['scaler_names'][:].astype('U13').tolist()
				dsic_index = scaler_names.index('ds_ic') #[index for index in scaler_names if index == 'ds_ic']
				DSIC = dat['MAPS']['scalers'][dsic_index][:]


				channels = dat['MAPS']['channel_names'][:].astype('U13').tolist()
				channels.append('XBIC')
				xrf.append(DSIC)

			return xvals, yvals, xrf, channels, summed_xrf

		# f = os.path.join(data_filepath, data_files[5])
		# with h5py.File(f, 'r') as dat:
		#     energy = dat['MAPS']['energy'][:]
		scans = list(scanlist.keys())
		channels = scanlist[scans[0]]['channels']

		files = os.listdir(self.tableofcontents['DataFolder'])
		scandat = {}
		for filename in files:
			if '2idd_' in filename:		#'2idd_' is specific to the output format from 2idd flyscan files, may need to change for other beamlines/file formats
				scan_num = str(int(filename[5:9]))

				if scan_num in scans:
					f = os.path.join(self.tableofcontents['DataFolder'], filename)
					x_data, y_data, xrf_data, all_channels, summed_xrf = read_2idd_h5(f)

					#get all xrf data, isolate the selected ones and save to dict
					xrfdat = {}
					for channel, xrf in zip(all_channels, xrf_data):
						if channel in channels:
							xrf=xrf[np.isnan(xrf)!=1]
							xrfdat[channel] = xrf

					#build dict with x, y, and xrf data for all scans
					scandat[int(scan_num)] = {
						'description': scanlist[scan_num]['description'],
						'x': x_data,
						'y': y_data,
						'xrf': xrfdat,
						'dwell': scanlist[scan_num]['dwell'],
						'energy': scanlist[scan_num]['energy'],
						'integratedxrf': summed_xrf
					}
		#sort scandat


		return scandat

	def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list(
			'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
			cmap(np.linspace(minval, maxval, n)))
		return new_cmap

	def plotxrf(self, scan, channel, x, y, xrf):
		savefolder = os.path.join(self.outputfolder, str(scan))
		if not os.path.exists(savefolder):
			os.mkdir(savefolder)

		image_format = 'jpeg'
		savepath = os.path.join(savefolder, channel + '.' + image_format)

		if not os.path.exists(savepath):
			xrf = np.array(xrf)
			xrf = xrf[:, :-2]	#remove last two lines, dead from 2idd flyscan

			color = cm.get_cmap('viridis')
			color_trimmed = self.truncate_colormap(cmap = color, minval = 0.0, maxval = 0.99)	#exclude highest brightness pixels to improve contrast with overlay text

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

	def plotoverview(self, scan, scandat):
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
		
		savefolder = os.path.join(self.outputfolder, str(scan))
		if not os.path.exists(savefolder):
			os.mkdir(savefolder)

		image_format = 'jpeg'
		savepath = os.path.join(savefolder, 'overviewmap.' + image_format)
		plt.savefig(savepath, format=image_format, dpi=300)
		plt.close() 

		return savepath

	def plotintegratedxrf(self, scan, scandat):
		xrf = scandat['integratedxrf']
		low = 1	#low energy plotting cutoff, keV
		high = float(scandat['energy']) + 0.5 #high energy plotting cutoff, keV

		# generate and save plot
		plt.figure(figsize = (8,2.5))
		plt.plot(
			xrf[(xrf[:,0]>=low) & (xrf[:,0]<=high),0],
			xrf[(xrf[:,0]>=low) & (xrf[:,0]<=high),1],
			color = 'k',
			linewidth = 1)
		ax = plt.gca()
		plt.yscale('log')
		plt.xlabel('Energy (keV)')
		plt.ylabel('Counts')
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
		
		savefolder = os.path.join(self.outputfolder, str(scan))
		if not os.path.exists(savefolder):
			os.mkdir(savefolder)

		image_format = 'jpeg'
		savepath = os.path.join(savefolder, 'integrated.' + image_format)
		
		plt.savefig(savepath, format=image_format, dpi=300, bbox_inches = 'tight')
		plt.close() 

		return savepath

	def makepdf(self):

		pdfmetrics.registerFont(TTFont('Open Sans', 'Vera.ttf'))

		PAGE_HEIGHT=defaultPageSize[1]; PAGE_WIDTH=defaultPageSize[0]
		styles = getSampleStyleSheet()



		doctitle = self.tableofcontents['Title']
		pageinfo = "2-ID-D, July 2019"
		titlefont = 'Open Sans'
		titlefontsize = 16
		bodyfont = 'Open Sans'
		bodyfontsize = 9

		def FirstPage(canvas, doc):
			canvas.saveState()
			frg_logo = 'C:\\Users\\RishiKumar\\Documents\\GitHub\\APS\\Sector 2\\Scripts\\Untitled.png'
			im = pilImage.open(frg_logo)
			imwidth, imheight = im.size
			logowidth = doc.width * 0.9
			logoheight = imheight * logowidth / imwidth

			canvas.drawImage(frg_logo, doc.leftMargin*2, doc.bottomMargin*3.5, logowidth, logoheight)

			# canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
			canvas.restoreState()

		def myLaterPages(canvas, doc):
			canvas.saveState()
			# canvas.setFont('Times-Roman',bodyfontsize)
			canvas.drawString(doc.leftMargin/2, doc.bottomMargin/2, "Page %d" % (doc.page))
			canvas.drawRightString(PAGE_WIDTH - doc.leftMargin/2, doc.bottomMargin/2, pageinfo)
			canvas.restoreState()

		def ScaledImage(filepath, dimension, size):
			im = pilImage.open(filepath)
			imwidth, imheight = im.size

			if dimension == 'width' or 'Width':
				printwidth = size
				printheight = size * imheight/imwidth
			elif dimension == 'height' or 'Height':
				printwidth = size * imwidth/imheight 
				printheight = size
			else:
				return #add error message here later

			image = Image(filepath, 
				width = printwidth,
				height = printheight,
				hAlign = 'CENTER')
			return image

		def generate_image_matrix(image_paths, max_num_cols, max_width, max_height):
			margin = 1.01

			# table_dimensions = {
			# 	1: (1,1),
			# 	2: (2,2),
			# 	3: (2,2),
			# 	4: (2,2),
			# 	5: (3,2),
			# 	6: (3,2),
			# 	7: (3,3),
			# 	8: (3,3),
			# 	9: (3,3),
			# 	10:(4,3),
			# 	11:(4,3),
			# 	12:(4,3),
			# 	13:(4,4),
			# 	15:(4,4),
			# 	16:(4,4)
			# 	}

			# num_cols, num_rows = table_dimensions[len(image_paths)]


			# if max_width/num_cols < max_height/num_rows:
			#     limiting_dimension = 'width'
			#     limiting_size = (max_width/num_cols) / margin
			# else:
			#     limiting_dimension = 'height'
			#     limiting_size = (max_height/num_rows) / margin     

			x = np.sqrt(len(image_paths))

			if x == np.floor(x):
				guess_num_cols = x
			else:       
				guess_num_cols = np.floor(x) + 1

			num_cols = int(guess_num_cols)
			num_rows = np.ceil(len(image_paths) / num_cols)

			if max_num_cols == 1:
				num_cols = 1
				num_rows = len(image_paths)
  

			limiting_size = max_width / num_cols

			if max_height/num_rows < limiting_size:
				limiting_dimension = 'height'
				limiting_size = max_height / num_rows
			else:
				limiting_dimension = 'width'
			limiting_size = limiting_size / margin

			filling = True
			img_matrix = []
			image_index = 0
			while filling:
				row_matrix = []
				for col in range(num_cols):
					if image_index < len(image_paths):
						im = ScaledImage(image_paths[image_index], dimension = limiting_dimension, size = limiting_size)
						row_matrix.append(im)
						image_index += 1
						if image_index == len(image_paths):
							filling = False
					else:
						row_matrix.append('')
				img_matrix.append(row_matrix)

			imtable = Table(img_matrix,
			 colWidths = limiting_size * margin,
			 rowHeights = limiting_size * margin)

			return imtable

		def build_scan_page(doc, Story, scan_number, title, text, overviewmap_image_filepath, scan_params, scan_image_filepaths):
			### text section
			headerstr = 'Scan ' + str(scan_number) + ': ' + title

			subheaderstr = []
			subheaderstr.append('\tArea: {0:d} x {1:d} um'.format(scan_params['x_range'], scan_params['y_range']))
			subheaderstr.append('\tStep Size: {0:s} um'.format(scan_params['stepsize']))
			subheaderstr.append('\tDwell Time: {0:s} ms'.format(scan_params['dwell']))
			subheaderstr.append('\tEnergy: {0:s} keV'.format(scan_params['energy']))


			Story.append(Paragraph(headerstr, styles['Heading1']))
			for each in subheaderstr:
				Story.append(Paragraph(each, styles['Heading3']))  
			Story.append(Paragraph(text, styles['Normal']))
			# Story.append(PageBreak())
			Story.append(FrameBreak())
			### overview map section

			imoverview = ScaledImage(overviewmap_image_filepath, 'width', doc.width * 0.45)
			Story.append(imoverview)
			Story.append(FrameBreak())
			# Story.append(PageBreak())

			###integrated xrf spectrum

			# imspectrum = ScaledImage(integratedspectrum_image_filepath, 'width', doc.width * 0.45)
			### xrf maps section
			# frame = doc.getFrame('xrfframe')
			# width = frame._aW
			# height = frame.aH

			imtable = generate_image_matrix(scan_image_filepaths,
				max_num_cols = 4,
				max_width = doc.width + doc.leftMargin,
				max_height = doc.height * 0.62)
			Story.append(imtable)
			# Story.append(PageBreak())

			return Story

		def build_title_page(doc, Story, title, subtitle):
			styles = getSampleStyleSheet()
			styles.add(ParagraphStyle(name='CenterTitle', fontSize = 24, alignment=TA_CENTER))
			styles.add(ParagraphStyle(name='CenterSubtitle', fontSize = 16, alignment=TA_CENTER, leading = 16))

			Story.append(Paragraph(title, styles['CenterTitle']))
			Story.append(FrameBreak())
			Story.append(Paragraph(subtitle, styles['CenterSubtitle']))
			Story.append(Paragraph(datetime.today().strftime('%Y-%m-%d'), styles['CenterSubtitle']))
			# Story.append(PageBreak())
			return(Story)

		def build_comparison_page(doc, Story, title, subtitle, comparisondict):
			margin = 0.99
			num_columns = len(comparisondict)



			Story.append(Paragraph(title, styles['Heading1']))
			Story.append(Paragraph(subtitle, styles['Normal']))


			#columns
			for _, vals in comparisondict.items():
				Story.append(FrameBreak())
				Story.append(Paragraph(vals['description'], styles['Normal']))
				Story.append(FrameBreak())
				imtable =  generate_image_matrix(vals['impaths'],
					max_num_cols = 1,
					max_width = doc.width / num_columns * margin,
					max_height = doc.height * 0.4)
				Story.append(imtable)

			return(Story)

		def go(doc, Story, outputpath):
			## title page template

			titleframe = Frame(
				x1 = doc.leftMargin,
				y1 = doc.height/2,
				width = doc.width,
				height = doc.height*.4,
				)

			subtitleframe = Frame(
				x1 = doc.leftMargin,
				y1 = doc.height/2 -doc.topMargin,
				width = doc.width,
				height = doc.height * 0.4,
				)

			## scan page template
			text_width = doc.width * 0.5
			text_height = doc.height * 0.35

			textframe = Frame(
				x1 = doc.leftMargin,
				y1 = PAGE_HEIGHT - doc.topMargin - text_height, 
				width = text_width,
				height = text_height,
				id = 'textframe')
			overviewmapframe = Frame(
				x1 = doc.leftMargin + doc.width * 0.5,
				y1 = PAGE_HEIGHT - doc.topMargin - text_height, 
				width = doc.width - text_width,
				height = text_height,
				id = 'overviewmapframe')
			xrfframe = Frame(
				x1 = doc.leftMargin * 0.5,
				y1 = doc.bottomMargin * 0.8, 
				width = doc.width + doc.leftMargin,
				height = doc.height - text_height,
				id = 'xrfframe')

			## 2-scan comparison page template

			def makecomparisontemplate(num_columns):
				margin = 1-0.01
				header_height = doc.height * 0.1 * margin
				subheader_height = doc.height * 0.1 * margin
				column_height = (doc.height - header_height - subheader_height) * margin
				column_width = (doc.width/ (num_columns)) * margin

				frames = []

				frames.append(Frame(
								x1 = doc.leftMargin,
								y1 = PAGE_HEIGHT - doc.topMargin - header_height, 
								width = doc.width,
								height = header_height,
								id = 'headerframe'))
				for n in range(num_columns):
					frames.append(Frame(
									x1 = doc.leftMargin + n*column_width,
									y1 = PAGE_HEIGHT - doc.topMargin - header_height - subheader_height, 
									width = column_width,
									height = subheader_height,
									id = 'subheader_' + str(n+1)))

					frames.append(Frame(
									x1 = doc.leftMargin + n*column_width,
									y1 = doc.bottomMargin, 
									width = column_width,
									height = doc.height - header_height - subheader_height,
									id = 'headerframe_' + str(n+1)))
				return frames


			doc.addPageTemplates([
								 PageTemplate(id='TitlePage',frames=[titleframe,subtitleframe], onPage = FirstPage),
								 PageTemplate(id='ScanPage',frames=[textframe,overviewmapframe,xrfframe], onPage = myLaterPages),
								 PageTemplate(id='Comparison_2',frames=makecomparisontemplate(2), onPage = myLaterPages),
								 PageTemplate(id='Comparison_3',frames=makecomparisontemplate(3), onPage = myLaterPages),
								 PageTemplate(id='Comparison_4',frames=makecomparisontemplate(4), onPage = myLaterPages)
								 ]
								)
			doc.build(Story)

		def report(self):

			def writescanlist(story, scanlist):
				scandat = self.readscans(scanlist = scanlist) #ghetto, maybe keep channels in upper level (assume constant for all scans in set, probably good assumption)			

				print('---scanlist----')
				keys = [int(x) for x in list(scandat.keys())]
				keys.sort()
				for key in keys:
					scan = key
					vals = scandat[key]

					impaths = []
					for channel, data in vals['xrf'].items():

						impaths.append(self.plotxrf(
										scan = scan,
										channel = channel,
										x = vals['x'],
										y = vals['y'],
										xrf = data
									 )
						)

					overviewpath = self.plotoverview(scan, scandat)
					integratedspectrumpath = self.plotintegratedxrf(scan, vals)

					scan_params = {'x_range': int(max(vals['x']) - min(vals['x'])),
								   'y_range': int(max(vals['y']) - min(vals['y'])),
								   'stepsize': str((max(vals['x'])-min(vals['x']))/(len(vals['x'])-1)),
								   'dwell': scanlist[str(scan)]['dwell'],
								   'energy': scanlist[str(scan)]['energy']
					}
					story.append(NextPageTemplate('ScanPage'))
					story.append(PageBreak())
					story = build_scan_page(
						doc = self.doc,
						Story = story,
						scan_number = scan,
						title = 'Scan ' + str(scan),
						text = vals['description'], 
						scan_params = scan_params,
						overviewmap_image_filepath = overviewpath, 
						scan_image_filepaths = impaths,
						)

				return story

			def writecomparison(story, comparison):
				for key, _ in comparison['scans'].items():
					comparison['scans'][key]['channels'] = comparison['channels']

				scandat = self.readscans(scanlist = comparison['scans']) #ghetto, maybe keep channels in upper level (assume constant for all scans in set, probably good assumption)			

				print('---comparison----')
				keys = [int(x) for x in list(scandat.keys())]
				keys.sort()

				channels = comparison['channels']
				for key in keys:
					comparison['scans'][str(key)]['impaths'] = []

					scan = key
					vals = scandat[key]

					for channel, data in vals['xrf'].items():

						comparison['scans'][str(key)]['impaths'].append(self.plotxrf(
										scan = scan,
										channel = channel,
										x = vals['x'],
										y = vals['y'],
										xrf = data
									 )
						)

				story = build_comparison_page(
					doc = self.doc,
					Story = story,
					title = comparison['title'],
					subtitle = comparison['description'], 
					comparisondict = comparison['scans'],
					)

				return story

			def writesection(story, section, contents):
				for key, content in contents.items():
					if key == 'scans':
						story = writescanlist(story, content)
					elif key == 'description':
						print('***TitlePage***')
						story.append(NextPageTemplate('TitlePage'))
						story.append(PageBreak())
						story = build_title_page(
							doc = self.doc,
							Story = self.Story,
							title = section,
							subtitle = content,
							)
					elif key == 'content':
						story = writesection(story, key, content)
					elif key == 'comparison':
						print("!!!Comparison!!!")
						num_columns = len(content['scans'])
						pagetemplate = 'Comparison_' + str(num_columns)	#only handles 2-4 columns

						story.append(NextPageTemplate(pagetemplate))
						story.append(PageBreak())

						story = writecomparison(
							story = self.Story,
							comparison = content)
					else:
						pass
				return story

			if not os.path.exists(self.outputfolder):
				os.mkdir(self.outputfolder)


			self.Story = build_title_page(
				doc = self.doc,
				Story = self.Story,
				title = self.tableofcontents['Title'],
				subtitle = self.tableofcontents['Description'],
				)

			for section, contents in self.tableofcontents['Contents'].items():
				self.Story = writesection(self.Story, section, contents)

			outputpath = os.path.join(self.outputfolder, self.title)
			go(self.doc, self.Story, outputpath)

			import subprocess
			subprocess.Popen(outputpath ,shell=True)

		report(self)




