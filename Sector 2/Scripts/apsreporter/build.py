from reportlab.platypus import SimpleDocTemplate, BaseDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, Frame, FrameBreak, PageTemplate, NextPageTemplate, KeepInFrame
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
import os
import json
import h5py
import numpy as np
from .plotting import truncate_colormap, plotxrf, plotoverview, plotintegratedxrf, plotcorrmat

ROOT_DIR = os.path.dirname(__file__)

class build:

	def __init__(self, tableofcontents, outputfolder, title, boundaries = False, pageinfo = "2-ID-D, July 2019", fontname = 'Open Sans', titlefontsize = 16, bodyfontsize = 9, logscale = False, overwrite = True):
		pdfmetrics.registerFont(TTFont(fontname, 'Vera.ttf'))
		self.PAGE_HEIGHT = defaultPageSize[1]
		self.PAGE_WIDTH = defaultPageSize[0]
		self.styles = getSampleStyleSheet()
		self.styles.add(ParagraphStyle(name='CenterTitle', fontSize = 24, alignment=TA_CENTER, leading = 24))
		self.styles.add(ParagraphStyle(name='CenterSubtitle', fontSize = 16, alignment=TA_CENTER, leading = 16))
		self.pageInfo = pageinfo
		self.title = title
		self.pageinfo = pageinfo
		self.titlefont = fontname
		self.titlefontsize = titlefontsize
		self.bodyfont = fontname
		self.bodyfontsize = bodyfontsize
		self.logscale = logscale
		self.overwrite = overwrite

		with open(tableofcontents, 'r') as f:
			self.tableofcontents = json.load(f)
		self.outputfolder = outputfolder
		f = os.path.join(outputfolder, title)
		self.doc = BaseDocTemplate(f,
			showBoundary = boundaries,
			leftMargin = inch/2,
			rightMargin = inch/2,
			bottomMargin = inch/2,
			topMargin = inch/2)
		self.Story = []

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
				USIC = dat['MAPS']['scalers'][scaler_names.index('us_ic')]


				channels = dat['MAPS']['channel_names'][:].astype('U13').tolist()
				channels.append('XBIC')
				channels.append('USIC')
				xrf.append(DSIC)
				xrf.append(USIC)

			return xvals, yvals, xrf, channels, summed_xrf

		# f = os.path.join(data_filepath, data_files[5])
		# with h5py.File(f, 'r') as dat:
		#     energy = dat['MAPS']['energy'][:]
		scans = list(scanlist.keys())
		channels = scanlist[scans[0]]['channels']
		ratiochannels = scanlist[scans[0]]['ratiochannels']

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
							npxrf =  np.array(xrf)
							xrfdat[channel] = npxrf[:,:-2] #remove last two lines, dead from flyscan

					#get the elemental ratio maps
					for ratioch in ratiochannels:
						elm1xrf = np.array(xrf_data[all_channels.index(ratioch[0])])[:,:-2]
						elm2xrf = np.array(xrf_data[all_channels.index(ratioch[1])])[:,:-2]
						ratiomap = np.divide(elm1xrf,elm2xrf)
						ratiochStr = ratioch[0] + ':' + ratioch[1]
						channels.append(ratiochStr)
						xrfdat[ratiochStr]=ratiomap

					#build dict with x, y, and xrf data for all scans
					scandat[int(scan_num)] = {
						'title': scanlist[scan_num]['title'],
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

	def FirstPage(self, canvas, doc):
		doc = self.doc
		canvas.saveState()
		frg_logo = os.path.join(ROOT_DIR, 'soleil_logo.png')
		im = pilImage.open(frg_logo)
		imwidth, imheight = im.size
		logowidth = doc.width * 0.9
		logoheight = imheight * logowidth / imwidth

		canvas.drawImage(frg_logo, doc.leftMargin*2, doc.bottomMargin*3.5, logowidth, logoheight)

		# canvas.drawString(inch, 0.75 * inch, "First Page / %s" % pageinfo)
		canvas.restoreState()

	def myLaterPages(self, canvas, doc):
		doc = self.doc
		canvas.saveState()
		# canvas.setFont('Times-Roman',bodyfontsize)
		canvas.drawString(doc.leftMargin/2, doc.bottomMargin/2, "Page %d" % (doc.page))
		canvas.drawRightString(self.PAGE_WIDTH - doc.leftMargin/2, doc.bottomMargin/2, self.pageinfo)
		canvas.restoreState()

	def get_frame_dimensions(self, pageid, frameid):
		doc = self.doc
		margin = 1
		page = [x for x in doc.pageTemplates if x.id == pageid]
		frame = [x for x in page[0].frames if x.id == frameid]
		width = frame[0].width * margin
		height = frame[0].height * margin
		return width, height

	def ScaledImage(self, filepath, dimension, size):
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

	def generate_image_matrix(self, image_paths, max_num_cols, max_width, max_height):
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
					im = self.ScaledImage(image_paths[image_index], dimension = limiting_dimension, size = limiting_size)
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

	def build_scan_page(self, scan_number, title, text, overviewmap_image_filepath, scan_params, scan_image_filepaths, integratedspectrum_image_filepath, corrmat_image_filepath):
		### text section
		headerstr = 'Scan ' + str(scan_number) + ': ' + title

		subheaderstr = []
		subheaderstr.append('\tArea: {0:d} x {1:d} um'.format(scan_params['x_range'], scan_params['y_range']))
		subheaderstr.append('\tStep Size: {0:s} um'.format(scan_params['stepsize']))
		subheaderstr.append('\tDwell Time: {0:s} ms'.format(scan_params['dwell']))
		subheaderstr.append('\tEnergy: {0:s} keV'.format(scan_params['energy']))


		self.Story.append(Paragraph(headerstr, self.styles['Heading1']))
		for each in subheaderstr:
			self.Story.append(Paragraph(each, self.styles['Normal']))  
		self.Story.append(Paragraph(text, self.styles['Normal']))
		# self.Story.append(PageBreak())
		self.Story.append(FrameBreak())
		### overview map section


		_, hlim = self.get_frame_dimensions('ScanPage', 'overviewmapframe')
		imoverview = self.ScaledImage(overviewmap_image_filepath, 'height', hlim)
		self.Story.append(imoverview)
		self.Story.append(FrameBreak())
		# self.Story.append(PageBreak())

		###integrated xrf spectrum
		wlim, hlim = self.get_frame_dimensions('ScanPage', 'intspecframe')
		imintspectrum = self.ScaledImage(integratedspectrum_image_filepath, 'width', wlim*0.9)
		self.Story.append(imintspectrum)
		self.Story.append(FrameBreak())			

		### correlation matrix
		wlim, hlim = self.get_frame_dimensions('ScanPage', 'corrmatframe')
		imcorrmat = self.ScaledImage(corrmat_image_filepath, 'height', hlim)
		self.Story.append(imcorrmat)
		self.Story.append(FrameBreak())

		# imspectrum = self.ScaledImage(integratedspectrum_image_filepath, 'width', doc.width * 0.45)
		### xrf maps section
		# frame = doc.getFrame('xrfframe')
		# width = frame._aW
		# height = frame.aH

		wlim, hlim = self.get_frame_dimensions('ScanPage', 'xrfframe')
		imtable = self.generate_image_matrix(scan_image_filepaths,
			max_num_cols = 4,
			max_width = wlim,
			max_height = hlim)
		self.Story.append(imtable)

	def build_title_page(self, title, subtitle):
		self.Story.append(Paragraph(title, self.styles['CenterTitle']))
		self.Story.append(FrameBreak())
		self.Story.append(Spacer(1, 0.25*inch))
		self.Story.append(Paragraph(subtitle, self.styles['CenterSubtitle']))
		self.Story.append(Spacer(1, 0.25*inch))
		self.Story.append(Paragraph(datetime.today().strftime('%Y-%m-%d'), self.styles['CenterSubtitle']))
		# self.Story.append(PageBreak())

	def build_comparison_page(self, title, subtitle, comparisondict):
		margin = 0.99
		num_columns = len(comparisondict)
		doc = self.doc



		self.Story.append(Paragraph(title, self.styles['Heading1']))
		self.Story.append(Paragraph(subtitle, self.styles['Normal']))


		#columns
		num_columns = len(comparisondict)
		wlim, hlim = self.get_frame_dimensions('Comparison_' + str(num_columns), 'imageframe_1')

		for _, vals in comparisondict.items():
			self.Story.append(FrameBreak())
			self.Story.append(Paragraph(vals['description'], self.styles['Normal']))
			self.Story.append(FrameBreak())
			imtable =  self.generate_image_matrix(vals['impaths'],
				max_num_cols = 1,
				max_width = wlim,
				max_height = hlim
				)
				# max_width = self.doc.width / num_columns * margin,
				# max_height = self.doc.height * 0.4)
			self.Story.append(imtable)

	def buildPageTemplates(self):
		## title page template
		doc = self.doc

		titleframe = Frame(
			x1 = doc.leftMargin,
			y1 = doc.height/2,
			width = doc.width,
			height = doc.height*.4,
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0
			)

		subtitleframe = Frame(
			x1 = doc.leftMargin,
			y1 = doc.height/2 -doc.topMargin,
			width = doc.width,
			height = doc.height * 0.4,
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0
			)

		## scan page template
		text_width = doc.width * 0.7
		text_height = doc.height * 0.15
		intspec_height = doc.height * 0.2
		intspec_width = doc.width * 0.7

		textframe = Frame(
			x1 = doc.leftMargin,
			y1 = self.PAGE_HEIGHT - doc.topMargin - text_height, 
			width = text_width,
			height = text_height,
			id = 'textframe',
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0)
		overviewmapframe = Frame(
			x1 = doc.leftMargin + text_width,
			y1 = self.PAGE_HEIGHT - doc.topMargin - text_height, 
			width = doc.width - text_width,
			height = text_height,
			id = 'overviewmapframe',
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0)
		intspecframe = Frame(
			x1 = doc.leftMargin ,
			y1 = self.PAGE_HEIGHT - doc.topMargin - text_height - intspec_height, 
			width = intspec_width,
			height = intspec_height,
			id = 'intspecframe',
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0)
		corrmatframe = Frame(
			x1 = doc.leftMargin + intspec_width,
			y1 = self.PAGE_HEIGHT - doc.topMargin - text_height - intspec_height, 
			width = doc.width - intspec_width,
			height = intspec_height,
			id = 'corrmatframe',
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0)
		xrfframe = Frame(
			x1 = doc.leftMargin * 0.5,
			y1 = doc.bottomMargin, 
			width = doc.width + doc.leftMargin,
			height = doc.height - text_height - intspec_height,
			id = 'xrfframe',
			leftPadding = 0,
			rightPadding = 0,
			topPadding = 0,
			bottomPadding = 0)

		## 2-scan comparison page template

		def makecomparisontemplate(num_columns):
			margin = 1-0.01
			header_height = doc.height * 0.1
			subheader_height = doc.height * 0.07
			column_height = (doc.height - header_height - subheader_height)
			column_width = (doc.width + doc.leftMargin) / (num_columns)

			frames = []

			frames.append(Frame(
							x1 = doc.leftMargin,
							y1 = self.PAGE_HEIGHT - doc.topMargin - header_height, 
							width = doc.width,
							height = header_height,
							id = 'headerframe',
							leftPadding = 0,
							rightPadding = 0,
							topPadding = 0,
							bottomPadding = 0))
			for n in range(num_columns):
				frames.append(Frame(
								x1 = doc.leftMargin/2 + n*column_width,
								y1 = self.PAGE_HEIGHT - doc.topMargin - header_height - subheader_height, 
								width = column_width,
								height = subheader_height,
								id = 'subheader_' + str(n),
								leftPadding = 0,
								rightPadding = 0,
								topPadding = 0,
								bottomPadding = 0))

				frames.append(Frame(
								x1 = doc.leftMargin/2 + n*column_width,
								y1 = doc.bottomMargin, 
								width = column_width,
								height = doc.height - header_height - subheader_height,
								id = 'imageframe_' + str(n),
								leftPadding = 0,
								rightPadding = 0,
								topPadding = 0,
								bottomPadding = 0))
			return frames


		doc.addPageTemplates([
							 PageTemplate(id='TitlePage',frames=[titleframe,subtitleframe], onPage = self.FirstPage),
							 PageTemplate(id='ScanPage',frames=[textframe,overviewmapframe,intspecframe,corrmatframe,xrfframe], onPage = self.myLaterPages),
							 PageTemplate(id='Comparison_2',frames=makecomparisontemplate(2), onPage = self.myLaterPages),
							 PageTemplate(id='Comparison_3',frames=makecomparisontemplate(3), onPage = self.myLaterPages),
							 PageTemplate(id='Comparison_4',frames=makecomparisontemplate(4), onPage = self.myLaterPages)
							 ]
							)

	def writescanlist(self, scanlist):
		scandat = self.readscans(scanlist = scanlist) #ghetto, maybe keep channels in upper level (assume constant for all scans in set, probably good assumption)			

		print('---scanlist----')
		keys = [int(x) for x in list(scandat.keys())]
		keys.sort()
		for key in keys:
			scan = key
			vals = scandat[key]

			impaths = []
			for channel, data in vals['xrf'].items():

				impaths.append(plotxrf(
								outputfolder = self.outputfolder,
								scan = scan,
								channel = channel,
								x = vals['x'],
								y = vals['y'],
								xrf = data,
								logscale = self.logscale,
								overwrite = self.overwrite
							 )
				)


			overviewpath = plotoverview(
				outputfolder = self.outputfolder,
				scan = scan,
				scandat = scandat)
			integratedspectrumpath = plotintegratedxrf(
				outputfolder = self.outputfolder, 
				scan = scan,
				scandat = vals)
			corrmatpath = plotcorrmat(
				outputfolder = self.outputfolder,
				scan = scan,
				scandat = vals)


			scan_params = {'x_range': int(max(vals['x']) - min(vals['x'])),
						   'y_range': int(max(vals['y']) - min(vals['y'])),
						   'stepsize': str((max(vals['x'])-min(vals['x']))/(len(vals['x'])-1)),
						   'dwell': scanlist[str(scan)]['dwell'],
						   'energy': scanlist[str(scan)]['energy']
			}
			self.Story.append(NextPageTemplate('ScanPage'))
			self.Story.append(PageBreak())
			self.build_scan_page(
				scan_number = scan,
				title = vals['title'],
				text = vals['description'], 
				scan_params = scan_params,
				overviewmap_image_filepath = overviewpath, 
				scan_image_filepaths = impaths,
				integratedspectrum_image_filepath = integratedspectrumpath,
				corrmat_image_filepath = corrmatpath
				)

	def writecomparison(self, comparison):
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

				comparison['scans'][str(key)]['impaths'].append(plotxrf(
								outputfolder = self.outputfolder,
								scan = scan,
								channel = channel,
								x = vals['x'],
								y = vals['y'],
								xrf = data
							 )
				)

		self.build_comparison_page(
			title = comparison['title'],
			subtitle = comparison['description'], 
			comparisondict = comparison['scans'],
			)

	def writesection(self, section, contents):
		for key, content in contents.items():
			if key == 'scans':
				self.writescanlist(scanlist = content)
			elif key == 'description':
				print('***TitlePage***')
				self.Story.append(NextPageTemplate('TitlePage'))
				self.Story.append(PageBreak())
				self.build_title_page(
					title = section,
					subtitle = content,
					)
			elif key == 'content':
				self.writesection(key, content)
			elif key == 'comparison':
				print("!!!Comparison!!!")
				num_columns = len(content['scans'])
				pagetemplate = 'Comparison_' + str(num_columns)	#only handles 2-4 columns

				self.Story.append(NextPageTemplate(pagetemplate))
				self.Story.append(PageBreak())

				self.writecomparison(
					comparison = content)
			else:
				pass

	def report(self):		

		if not os.path.exists(self.outputfolder):
			os.mkdir(self.outputfolder)

		self.buildPageTemplates()

		self.build_title_page(
			title = self.tableofcontents['Title'],
			subtitle = self.tableofcontents['Description'],
			)

		for section, contents in self.tableofcontents['Contents'].items():
			self.writesection(section, contents)

		outputpath = os.path.join(self.outputfolder, self.title)
		
		self.doc.build(self.Story)
		return outputpath