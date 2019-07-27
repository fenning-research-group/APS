from datetime import datetime
from tkinter import Tk, IntVar, Checkbutton, Button, mainloop, W
import h5py
import numpy as np
import json
import pandas as pd
import csv
import os

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
		def setchannels(self, channels = None):
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
					all_channels = data['MAPS']['channel_names'][:].astype('U13').tolist()

				all_channels.append('XBIC')	#this option links to the downstream ion chamber scaler, typically used for recording XBIC current
				self._channels = pick_channels(all_channels)

		# def description(self, scan, description):
		# 	if scan in self._scans:
		# 		self._scans[scan] = description
		# 	else:
		# 		print('Scan %d not in scan list.'.format(scan))
