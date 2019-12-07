import numpy as np
import matplotlib.pyplot as plt
from readMDA import readMDA
import h5py
import os
import multiprocessing as mp
import cv2 
from tqdm import tqdm
import time
import json



### H5 processing scripts

def generate_energy_list(cal_offset = -0.0151744, cal_slope = 0.0103725, cal_quad = 0.00000):
        energy = [cal_offset + cal_slope*x + cal_quad*x*x for x in range(2048)]
        return energy

def LoadMDA(scannum, mdadirectory, imagedirectory, only3d = False):   
	print('Reading MDA File')  
	for f in os.listdir(mdadirectory):
			if int(f.split('SOFT_')[1][:-4]) == scannum:
					mdapath = os.path.join(mdadirectory, f)
					break

	if only3d:
			data = readMDA(mdapath, verbose=0, maxdim = 3)
	else:
			data = readMDA(mdapath, verbose=0, maxdim = 2)
	ndim = len(data)-1 if data[0]["dimensions"][-1] != 2048 else len(data)-2
	image_path = os.path.join(imagedirectory, str(scannum))
	image_list = [imagefile.name for imagefile in os.scandir(image_path) if imagefile.name.endswith('.tif')]
	image_index = np.array([int(filename.split(".")[0].split("_")[-1]) for filename in image_list])
	image_list = np.take(image_list , image_index.argsort())
	image_index.sort()


	nbin = 1
	mda_index = 0
	if ndim == 2:
			nfile = data[0]['dimensions'][0] * data[0]['dimensions'][1]
			for i in range(data[ndim].nd):
					dname = data[ndim].d[i].name
					if "FileNumber" in dname:
							mda_index = np.array(data[2].d[i].data)
							print (mda_index.max(), image_index.max(), mda_index.min(), image_index.min())
	else:
			nfile = data[0]['dimensions'][0]
	
	positioners = (data[1].p[0].name, data[2].p[0].name)

	output = {
			'scan': scannum,
			'ndim': ndim,
			'positioners': {
					'names': positioners,
					'values': [data[1].p[0].data, data[2].p[0].data[0]]     #two lists, first = positioner 1 values, second = positioner 2 values
					},
			# 'logbook': readlogbook(scannum),
			'image_list': image_list,
			'image_index': image_index,
			'mda_index': mda_index
	}

	if only3d:
			xrfraw = {}
			for d in data[3].d:
					name = d.name.split(':')[1].split('.')[0]
					xrfraw[name] = {
									'energy': generate_energy_list(),
									'counts': np.array(d.data)
							}
			output['xrfraw'] = xrfraw

	return output

def _MDADataToH5(data, h5directory, Image_folder, twothetaccdpath, gammaccdpath, loadimages = True):
	p = mp.Pool(mp.cpu_count())
	# p = mp.Pool(4)
	filepath = os.path.join(h5directory, 'scan_{0}.hdf5'.format(data['scan']))

	with h5py.File(filepath, 'w') as f:
			
			info = f.create_group('/info')
			info.attrs['description'] = 'Metadata describing scan parameters, sample, datetime, etc.'
			temp = info.create_dataset('scan', data = data['scan'])
			temp.attrs['description'] = 'Scan number'
			temp = info.create_dataset('ndim', data = data['ndim'])
			temp.attrs['description'] = 'Number of dimensions in scan dataset'

			dimages = f.create_group('/xrd/im')
			dimages.attrs['description'] = 'Contains diffraction detector images.'
			dpatterns = f.create_group('/xrd/pat')
			dpatterns.attrs['description'] = 'Contains diffraction data, collapsed to twotheta vs counts. Note that twotheta values depend on incident beam energy!'

			xrf = f.create_group('/xrf')
			xrf.attrs['description'] = 'Contains fluorescence data from both single element (mca8) and four-element (mca0-3) detectors.'

			detectorlist = [x.encode('utf-8') for x in data['xrfraw'].keys()]
			detectorname = xrf.create_dataset('names', data = detectorlist)
			detectorname.attrs['description'] = 'Names of detectors. mca8 = single element, mca0-3 = individual elements on 4 element detector'

			energydata = np.array([np.array(x['energy']) for _,x in data['xrfraw'].items()])
			energy = xrf.create_dataset('e', data = energydata)
			energy.attrs['description'] = 'Energy scale for each detector, based on cal_offset, cal_slope, and cal_quad provided during file compilation.'

			xrfcounts = xrf.create_dataset('cts', data = np.array([x['counts'] for _,x in data['xrfraw'].items()]), chunks = True, compression = "gzip")
			xrfcounts.attrs['description'] = 'Array of 2-d arrays of counts, for each detector at each scan point (numdet * xpts * ypts * 2048)'
			
			intxrfcounts = xrf.create_dataset('intcts', data = np.sum(np.sum(xrfcounts, axis = 1), axis = 1))
			intxrfcounts.attrs['description'] = 'Array of area-integrated fluorescence counts for each detector'
			
			f.flush()	# write xrf data to disk

			if loadimages:
					numpts = 200
					twothetaimage = dimages.create_dataset('twotheta', data = np.genfromtxt(twothetaccdpath, delimiter=','))
					twothetaimage.attrs['description'] = 'Map correlating two-theta values to each pixel on diffraction ccd'
					temp = dimages.create_dataset('gamma', data = np.genfromtxt(gammaccdpath, delimiter=','))
					temp.attrs['description'] = 'Map correlating gamma values to each pixel on diffraction ccd'
					tolerance = (twothetaimage[:].max()-twothetaimage[:].min()) / numpts
					interp_twotheta = dpatterns.create_dataset('twotheta', data = np.linspace(twothetaimage[:].min()+tolerance/2, twothetaimage[:].max()-tolerance/2, numpts))
					interp_twotheta.attrs['description'] = 'Twotheta values onto which ccd pixel intensities are collapsed.'
					imnums = data['mda_index']-1      #files are saved offset by 1 for some reason
					xrdcounts = dpatterns.create_dataset('cts', data = np.zeros((imnums.shape[0], imnums.shape[1], numpts)), chunks = True)
					xrdcounts.attrs['description'] = 'Collapsed diffraction counts for each scan point.'
					intxrdcounts = dpatterns.create_dataset('intcts', data = np.zeros((numpts,)))
					intxrdcounts.attrs['description'] = 'Collapsed, area-integrated diffraction counts.'
					imgpaths = [os.path.join(Image_folder, str(data['scan']), 'scan_{0}_img_Pilatus_{1}.tif'.format(data['scan'], int(x))) for x in imnums.ravel()]
					print('Loading Images')
					imgdata = p.starmap(cv2.imread, [(x, -1) for x in imgpaths])
					d = imgdata[0].shape
					imgdata = np.array(imgdata).reshape(imnums.shape[0], imnums.shape[1], d[0], d[1])

					images = dimages.create_dataset('ccd', data = imgdata, compression = 'gzip', chunks = True)
					print('Fitting twotheta')

					f.flush()
					ttmask = [np.abs(twothetaimage - tt_) <= tolerance for tt_ in interp_twotheta]

					for m,n in tqdm(np.ndindex(imnums.shape), total = imnums.shape[0] * imnums.shape[1]):
						# for tidx, tt in enumerate(interp_twotheta):
							# im = imgdata[m,n]
							# xrdcounts[m,n,tidx] = np.sum(im[np.abs(twothetaimage[:]-tt) <= tolerance])	#add diffraction from all points where twotheta falls within tolerance
						xrdcounts[m,n] = p.starmap(np.sum, [(imgdata[m,n][ttmask_],) for ttmask_ in ttmask])
					intxrdcounts[()] = xrdcounts[()].sum(axis=0).sum(axis=0)	#sum across map dimensions

					# images = None
					# for m, n in np.ndindex(imnums.shape):
					#         impath = os.path.join(Image_folder, str(data['scan']), 'scan_{0}_img_Pilatus_{1}.tif'.format(data['scan'], int(imnums[m,n])))
					#         im = cv2.imread(impath, -1)
					#         if images is None:
					#                 images = dimages.create_dataset('ccd', (imnums.shape[0], imnums.shape[1], im.shape[0], im.shape[1]), compression = "gzip", chunks = True)
					#                 images.attrs['description'] = 'Raw ccd images for each scan point.'
					#                 # images = [[None for n in range(imnums.shape[1])] for m in range(imnums.shape[0])]
					#         images[m,n,:,:] = im
					#         for tidx, tt in enumerate(interp_twotheta):
					#                 xrdcounts[m,n,tidx] = np.sum(im[np.abs(twothetaimage[:]-tt) <= tolerance])
					#                 intxrdcounts = intxrdcounts + xrdcounts[m,n,tidx]
	p.close()

class H5Daemon():
	def __init__(self, rootdirectory):
		self.rootDirectory = rootdirectory
		self.mdaDirectory = os.path.join(self.rootDirectory, 'mda')
		self.h5Directory = os.path.join(self.rootDirectory, 'h5')
		if not os.path.isdir(self.h5Directory):
			os.mkdir(self.h5Directory)

		self.logDirectory = os.path.join(self.rootDirectory, 'Logging')
		self.qmatDirectory = os.path.join(self.logDirectory, 'qmat')
		self.imageDirectory = os.path.join(self.rootDirectory, 'Images')

	def MDAToH5(self, scannum = None):
		print('=== Processing Scan {0} from MDA to H5 ==='.format(scannum))
		data = LoadMDA(scannum, self.mdaDirectory, self.imageDirectory, only3d = True)
		_MDADataToH5(
			data,
			self.h5Directory,
			self.imageDirectory,
			os.path.join(self.qmatDirectory, 'twotheta.csv'),
			os.path.join(self.qmatDirectory, 'gamma.csv'),
			self.loadimages = True
			)


	def Listener(self, functions = ['scan2d']):
		import epics
		import epics.devices
		
		def findMostRecentScan():
			fids = os.listdir(self.mdaDirectory)
			scannums = [int(x.split('SOFT_')[1].split('.mda')[0]) for x in fids]
			return max(scannums)
		def lookupScanFunction(scannum):
			with open(os.path.join(self.logDirectory, 'verboselog.json')) as f:
				logdata = json.load(f)
			return f[scannum]['ScanFunction']

		self.lastProcessedScan = 0
		while True:	#keep running unless manually quit by ctrl-c
			mostRecentScan = findMostRecentScan()	#get most recent scan number
			if self.lastProcessedScan < findMostRecentScan #have we looked at this scan yet?
				scanFunction = lookupScanFunction(mostRecentScan)	#if not, lets see if its a scan we want to convert to an h5 file
				if scanFunction in functions:	#if it is one of our target scan types (currently only works on scan2d as of 20191206)
					if epics.caget("26idc:filter:Fi1:Set") == 0:	#make sure that the scan has completed (currently using filter one being closed as indicator of completed scan)
						self.MDAToH5(scannum = mostRecentScan)	#if we passed all of that, fit the dataset
				else:
					self.lastProcessedScan = mostRecentScan 	#if the scan isnt a fittable type, set the scan number so we dont look at it again

			time.sleep(5)	# check for new files every 5 seconds