import sys
import epics
import epics.devices
import time
import datetime
import numpy as np
import os 
import math
import socket

class Logger():
	"""
	Object to handle logging of motor positions, filter statuses, and XRF ROI assignments per scan. 

	REK 20191206
	"""
	def __init__(self, sample = '', note = ''):
		self.rootDir = os.path.join(
			epics.caget(scanrecord+':saveData_fileSystem',as_string=True),	#file directory
			epics.caget(scanrecord+':saveData_subDir',as_string=True)		#subdir for our data
			)
		self.logDir = os.path.join(self.rootDir, 'Logging')
		if not os.path.isdir(self.logDir):
			os.mkdir(self.logDir)

		self.logFilepath = os.path.join(self.logDir, 'verboselog.json')
		if not os.path.exist(self.logFilepath):
			with open(self.logFilepath, 'w') as f:
				json.dump({}, f)    #intialize as empty dictionary

		self.sample = sample    #current sample being measured
		self.note = note

		self.motorDict = {  #array of motor labels + epics addresses
					"fomx": '26idcnpi:m10.VAL',
					"fomy": '26idcnpi:m11.VAL',
					"fomz": '26idcnpi:m12.VAL',
					# "samx": '26idcnpi:m16.VAL',
					"samy": '26idcnpi:m17.VAL',
					# "samz": '26idcnpi:m18.VAL',
					"samth": 'atto2:PIC867:1:m1.VAL',
					"osax": '26idcnpi:m13.VAL',
					"osay": '26idcnpi:m14.VAL',
					"osaz": '26idcnpi:m15.VAL',
					"condx": '26idcnpi:m5.VAL',
					"attox": 'atto2:m3.VAL',
					"attoz": '26idcNES:sm27.VAL',
					"samchi": 'atto2:m1.VAL',
					"samphi": 'atto2:m2.VAL',
					"objx": '26idcnpi:m1.VAL',
					"xrfx": '26idcDET:m7.VAL',
					# "piezox": '26idcSOFT:sm1.VAL',
					# "piezoy": '26idcSOFT:sm2.VAL',
					"hybridx": '26idcnpi:X_HYBRID_SP.VAL',
					"hybridy": '26idcnpi:Y_HYBRID_SP.VAL',
					"twotheta": '26idcSOFT:sm3.VAL',
					"gamma":    '26idcSOFT:sm4.VAL',
					"filter1": "26idc:filter:Fi1:Set",  
					"filter2": "26idc:filter:Fi2:Set",
					"filter3": "26idc:filter:Fi3:Set",
					"filter4": "26idc:filter:Fi4:Set",
					"energy": "26idbDCM:sm8.RBV",	
				}

	def getXRFROI(self):
		"""
		loads ROI assignments from MCA1, assumes we are using same ROI assignments for MCA 1-4
		"""
		ROI = []
		for roinum in range(32):
			ROI.append({
				'Line': epics.caget('26idcXMAP:mca1.R{0}NM'.format(roinum)),
				'Low': epics.caget('26idcXMAP:mca1.R{0}LO'.format(roinum)),
				'High': epics.caget('26idcXMAP:mca1.R{0}HI'.format(roinum))
				})

		return ROI

	def updateLog(self, scanFunction, scanArgs):
		self.scanNumber = epics.caget(scanrecord+':saveData_scanNumber',as_string=True)

		self.scanEntry = {
			'ROIs': self.getXRFROI(),
			'Sample': self.sample,
			'Note': self.note,
			'Date': str(datetime.datetime.now().date()),
			'Time': str(datetime.datetime.now().time()),
			'ScanFunction': scanFunction
			'ScanArgs': scanArgs
		}
		
		for label, key in motordict.items():
			self.scanEntry[label] = epics.caget(key, as_string = True)
		
		### Add entry to log file
		with open(self.logbook_filepath, 'r') as f:
			fullLogbook = json.load(f)
		fullLogbook[scanNumber] = self.scanEntry
		with open(self.logbook_filepath, 'w') as f:
			json.dump(fullLogbook, f)


		self.lastScan
