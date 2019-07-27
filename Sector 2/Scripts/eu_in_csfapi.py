import sys
sys.path.append('/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV')

from apsreporter import tableofcontents, section, scanlist, comparison, build
import os

datafolder = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV'
outputfolder = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/1%Eu_noTreat_6pt98keV'
pdf_filename = '1pctEuCsFAPI.pdf'


def toc():
	t = tableofcontents(
		title = 'Europium Doping in CsFAPbI3',
		description = 'CsFAPI3 devices with and without 1% Eu loading undergo stress testing by heat, light soaking at Voc, and maximum power point tracking.',
		outputfile = os.path.join(outputfolder, '1pctEuCsFAPI_tableofcontents.json'),
		datafolder = datafolder
		)

	t.energy = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/scanenergies.csv'
	t.dwell = '/Volumes/GoogleDrive/My Drive/APS/APS2019c2_2idd/XRF/h5RawData/dwelltimes.csv'

	s1 = section(
		title = '1\% Eu in CSFAPbI3',
		description = ' '
	)

	sl1 = scanlist(
		datafolder = datafolder,
		scans = {
			# 277: ' ',
			# 278: ' ',
			# 279: 'Cesium precipitates',
			# 289: 'Map at Eu 2+ absorption edge energy',
			# 290: 'Map at Eu 3+ absorption edge energy',
			291: 'Smaller overview map',
			292: 'Map showing clear europium content at Pb/I boundaries',
			293: 'Map showing clear europium content at Pb/I boundaries',
		}
	)

	# sl1 = scanlist(
	# 	datafolder = 'G:\My Drive\FRG\Projects\APS\\2IDD_2019\Sample Data - 150C HEP\\data',
	# 	scans = {
	# 		63: 'scan 63 description',
	# 		64: 'scan 64 description',
	# 		65: 'scan 65 description'
	# 	}
	# )

	print('scan list 1')
	sl1.setchannels()

	c1 = comparison(
		title = 'Testing compaarison component',
		description = 'Comparison 1 Description',
		datafolder = datafolder,
		scans = {
			291: 'Smaller overview map',
			292: 'Map showing clear europium content at Pb/I boundaries',
			293: 'Map showing clear europium content at Pb/I boundaries',
		}
	)
	c1.setchannels()


	s1.contents=('comparison1',c1)
	s1.contents = ('scans1', sl1)
	# s1.contents = ('comparison1', c1)

	# s2 = section(
	# 	title = 'Section 2',
	# 	description = 'S2 desc'
	# )
	# s2.contents = ('scans1', sl1)

	t.contents = ('1\% Eu, no stress testing', s1)
	# t.contents = ('150C HEP 2', s2)

	t.show()
	f = t.export()

def pdf():
	b = build(
		tableofcontents = os.path.join(outputfolder, '1pctEuCsFAPI_tableofcontents.json'),
		outputfolder = os.path.join(outputfolder, 'output'),
		title =pdf_filename,
		boundaries = 0,
		overwrite = True,
		logscale = False
		)

	b.report()

	import subprocess
	subprocess.Popen(os.path.join(outputfolder, 'output', pdf_filename) ,shell=True)

toc()
pdf()
