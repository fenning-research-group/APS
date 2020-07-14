import numpy as np
import matplotlib.pyplot as plt
from .readMDA import readMDA
import h5py
import os
import multiprocessing as mp
import cv2 
from tqdm import tqdm
import time
import json
from matplotlib.colors import LogNorm


### scripts for working with Daemon-generated H5 Files

def diffraction_map(fpath, twotheta = None, q = None, ax = None, tol = 2):
	"""
	Plots maps of diffraction across map area
	"""
	colors = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges]

	if ax is None:
		fig, ax = plt.subplots()
		displayPlot = True
	else:
		displayPlot = False

	if twotheta is not None:
		x = 'twotheta'
		xdata = twotheta
		xlabel = '$\degree$'
	elif q is not None:
		x = 'q'
		xdata = q
		xlabel = ' $A^{-1}$'
	else:
		print('Error: Provide either twotheta or q values!')
		return

	with h5py.File(fpath, 'r') as d:
		x0 = d['xrd']['pat'][x][()]
		diffmap = d['xrd']['pat']['cts'][()]

	# plot results
	alpha = 1/len(xdata)
	for i_, x_, c_ in zip(range(len(xdata)),xdata[::-1], colors):
		idx = np.argmin(np.abs(x0 - x_))
		ax.imshow(diffmap[:,:,idx-tol:idx+tol].sum(axis = 2), alpha = alpha, cmap = c_)
		ax.text(1.01, 0.99 - 0.05*i_, '{0}{1}'.format(x_, xlabel), color = c_(180), ha = 'left', va = 'top', transform = ax.transAxes)
	if displayPlot:
		plt.show()

### convenience functions

def twotheta_to_q(twotheta, energy = None):
	"""
	Converts a twotheta diffraction angle to scattering q vector magnitude given 
	the incident photon energy in keV
	"""

	if energy is None:
		print('No photon energy provided by user - assuming 8.040 keV (Cu-k-alpha)')
		energy = 8.040
	wavelength = 12.398/energy

	return 	(4*np.pi/wavelength)*np.sin(np.deg2rad(twotheta/2))

def q_to_twotheta(q, energy = None):
	"""
	Converts a scattering q vector magnitude to twotheta diffraction angle given 
	the incident photon energy in keV
	"""

	if energy is None:
		print('No photon energy provided by user - assuming 8.040 keV (Cu-k-alpha)')
		energy = 8.040
	wavelength = 12.398/energy

	return 2*np.rad2deg(np.arcsin((q*wavelength)/(4*np.pi)))

def twotheta_adjust(twotheta, e, e0 = None):
	'''
	converts twotheta value from initial x-ray energy (defaults to Cu-ka at 8.04 keV)
	to that at another x-ray energy

		twotheta: twotheta angle (degrees) at initial energy
		e: energy to adjust angle to (keV)
		e0: energy to adjust angle from (keV)
	'''

	if e0 is None:
		print('No initial photon energy provided by user - assuming 8.040 keV (Cu-k-alpha)')
		e0 = 8.040

	return q_to_twotheta(
				q = twotheta_to_q(twotheta, energy = e0),
				energy = e
				)

### H5 processing Daemon + associated scripts

class Daemon():
	def __init__(self, rootdirectory, functions = ['scan2d']):
		self.rootDirectory = rootdirectory
		self.mdaDirectory = os.path.join(self.rootDirectory, 'mda')
		self.h5Directory = os.path.join(self.rootDirectory, 'h5')
		if not os.path.isdir(self.h5Directory):
			os.mkdir(self.h5Directory)

		self.logDirectory = os.path.join(self.rootDirectory, 'Logging')
		if not os.path.isdir(self.logDirectory):
			os.mkdir(self.logDirectory)
			
		self.qmatDirectory = os.path.join(self.logDirectory, 'qmat')
		if not os.path.isdir(self.qmatDirectory):
			os.mkdir(self.qmatDirectory)
			print('Make sure to save qmat files to {}'.format(self.qmatDirectory))
		# with open(os.path.join(self.qmatDirectory, 'qmat.json'), 'r') as f:
		# 	self.qmat = json.load(f)
		self.imageDirectory = os.path.join(self.rootDirectory, 'Images')

		self.Listener() #start the daemon

	def MDAToH5(self, scannum = None, loadimages = True):
		print('=== Processing Scan {0} from MDA to H5 ==='.format(scannum))
		data = load_MDA(scannum, self.mdaDirectory, self.imageDirectory, self.logDirectory, only3d = True)
		_MDADataToH5(
			data,
			self.h5Directory,
			self.imageDirectory,
			os.path.join(self.qmatDirectory, 'twotheta.csv'),
			os.path.join(self.qmatDirectory, 'gamma.csv'),
			loadimages = loadimages
			)

	def Listener(self, functions):
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
			if self.lastProcessedScan < findMostRecentScan: #have we looked at this scan yet?
				scanFunction = lookupScanFunction(mostRecentScan)	#if not, lets see if its a scan we want to convert to an h5 file
				if scanFunction in functions:	#if it is one of our target scan types (currently only works on scan2d as of 20191206)
					if epics.caget("26idc:filter:Fi1:Set") == 0:	#make sure that the scan has completed (currently using filter one being closed as indicator of completed scan)
						try:
							self.MDAToH5(scannum = mostRecentScan)	#if we passed all of that, fit the dataset
						except:
							print('  Error converting scan {} to H5'.format(mostRecentScan))
						self.lastProcessedScan = mostRecentScan
				else:
					self.lastProcessedScan = mostRecentScan 	#if the scan isnt a fittable type, set the scan number so we dont look at it again

			time.sleep(5)	# check for new files every 5 seconds


	### Plotting functions - should be moved outside of Daemon object

	def TwoThetaWaterfall(self, scannum, numtt = 200, timestep = 1, xrdlib = [], hotccdthreshold = np.inf, ax = None):
		plotAtTheEnd = False
		if ax is None:
			fig, ax = plt.subplots(figsize = (8, 4))
			plotAtTheEnd = True

		with open(os.path.join(self.qmatDirectory, 'qmat.json'), 'r') as f:
			qmat = json.load(f)

		imdir = os.path.join(self.imageDirectory, str(scannum))
		imfids = [os.path.join(imdir, x) for x in os.listdir(imdir) if 'Pilatus' in x]	#currently only anticipates Pilatus CCD images

		tt = np.linspace(qmat['twotheta'].min(), qmat['twotheta'].max(), numtt)
		cts = np.full((len(imfids), numtt), np.nan)
		time = np.linspace(0, len(imfids))*timestep
		
		for idx, fid in tqdm(enumerate(imfids), total = len(imfids), desc = 'Loading Images'):
			im = np.asarray(PIL.Image.open(fid))
			if im.max() >= hotccdthreshold:
				pass
			else:
				for ttidx, tt_ in enumerate(tt):
					mask = np.abs(qmat['twotheta'] - tt_) <= 0.05
					cts[idx,ttidx] = im[mask].sum()
		
		im = ax.imshow(cts, cmap = plt.cm.inferno, extent = [tt[0], tt[-1], time[0], time[-1]], norm = LogNorm(1, np.nanmax(cts))) #aspect = 0.02, 
		ax.set_aspect('auto')
		ax.set_xlabel('$2\Theta\ (\degree,10keV)$')
		ax.set_ylabel('Time (s)')
		cb = plt.colorbar(im, ax = ax, fraction = 0.03)
		cb.set_label('Counts (log scale)')
		ticksize = time.max()/20
		
		for idx, xlib_ in enumerate(xrdlib):
			c = plt.cm.tab10(idx)
			ax.text(1.0, 0.6 - idx*0.05, xlib_['title'], color = c, transform = fig.transFigure)        
			for p in xlib_['peaks']:
				if p <= tt.max() and p >= tt.min():
					ax.plot([p, p], [time[-1] + (0.5*idx)*ticksize, time[-1] + (0.5*(idx) + 0.8) * ticksize], color = c, linewidth = 0.6, clip_on = False)
		ax.set_clip_on(False)
		ax.set_ylim((0, time.max()))

		if plotAtTheEnd:
			plt.plot()

	def SumCCD(self, scannum, numtt = 200, xrdlib = [], hotccdthreshold = np.inf, ax = None):
		plotAtTheEnd = False
		if ax is None:
			fig, ax = plt.subplots(figsize = (8, 4))
			plotAtTheEnd = True
		elif len(ax) != 2:
			print('Error: If providing axes to plot to, a list of two axes must be provided! Aborting.')
			return

		with open(os.path.join(self.qmatDirectory, 'qmat.json'), 'r') as f:
			qmat = json.load(f)

		imdir = os.path.join(rootdir, 'Images', str(scannum))
		imfids = [os.path.join(imdir, x) for x in os.listdir(imdir) if 'Pilatus' in x] #currently only anticipates Pilatus images
		
		for idx, fid in tqdm(enumerate(imfids), total = len(imfids), desc = 'Loading Images'):
			im = np.asarray(PIL.Image.open(fid))
			if idx == 0:
				ccdsum = np.zeros(im.shape)
			if im.max() < hotccdthreshold:
				ccdsum += im
		
		ax[0].imshow(ccdsum, cmap = plt.cm.gray, norm = LogNorm(0.1, ccdsum.max()))
		
		tt = np.linspace(qmat['twotheta'].min(), qmat['twotheta'].max(), numtt)
		cts = []
		for idx, tt_ in enumerate(tt):
			mask = np.abs(qmat['twotheta'] - tt_) <= 0.05
			cts.append(ccdsum[mask].sum())
		ax[1].plot(tt, cts)
		xlim0 = ax[1].get_xlim()
		ylim0 = ax[1].get_ylim()
		
		for idx, xlib_ in tqdm(enumerate(xrdlib), total = len(xrdlib), desc = 'Fitting'):
			c = plt.cm.tab10(idx)
			# ax[0].text(1.0, 1.0 - idx*0.05, xlib_['title'], color = c, transform = fig.transFigure) 
			cmap = colors.ListedColormap([c, c])
			bounds=[0,1,10]
			norm = colors.BoundaryNorm(bounds, cmap.N)
			
			first = True
			for p in xlib_['peaks']:            
				mask = np.abs(qmat['twotheta'] - p) <= 0.05
				mask = mask.astype(float)*5
				mask[mask == 0] = np.nan
	#             cmask = cmask.astype(float)
	#             cmask[cmask == 0] = np.nan
	#             mask = np.array([mask*c_ for c_ in c]).reshape(195, 487, 4)
	#             return mask
	#             mask[mask == 0] = np.nan
				ax[0].imshow(mask, cmap = cmap, alpha = 0.4)#, cmap = plt.cm.Reds)
				if first:
					ax[1].plot(np.ones((2,))*p, ax[1].get_ylim(), label = xlib_['title'], color = c, linewidth = 0.3, linestyle = ':')
					first = False
				else:
					ax[1].plot(np.ones((2,))*p, ax[1].get_ylim(), color = c, linewidth = 1, linestyle = ':')
	   
		ax[1].set_xlim(xlim0)
		ax[1].set_ylim(ylim0)
		leg = ax[1].legend(loc = 'upper right')
		for line in leg.get_lines():
			line.set_linewidth(2.0)
			line.set_linestyle('-')

		ax[1].set_xlabel('$2\Theta\ (10keV)$')
		ax[1].set_ylabel('Counts')
		ax[0].set_title('Integrated Diffraction, Scan {0}'.format(scannum))
		
		if plotAtTheEnd:
			plt.show()


def generate_energy_list(cal_offset = -0.0151744, cal_slope = 0.0103725, cal_quad = 0.00000):
	energy = [cal_offset + cal_slope*x + cal_quad*x*x for x in range(2048)]
	return energy

def load_MDA(scannum, mdadirectory, imagedirectory, logdirectory, only3d = False):   
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
			with open(os.path.join(logdirectory, 'mcacal.json'), 'r') as f:
				mcacal = json.load(f)
			xrfraw = {}
			for d in data[3].d:
					name = d.name.split(':')[1].split('.')[0]
					xrfraw[name] = {
									'energy': generate_energy_list(*mcacal[name]),
									'counts': np.array(d.data)
							}
			output['xrfraw'] = xrfraw

	return output

def _MDADataToH5(data, h5directory, imagedirectory, twothetaccdpath, gammaccdpath, loadimages = True):
	p = mp.Pool(mp.cpu_count())
	# p = mp.Pool(4)
	filepath = os.path.join(h5directory, '26idbSOFT_{0:04d}.h5'.format(data['scan']))
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
					imgpaths = [os.path.join(imagedirectory, str(data['scan']), 'scan_{0}_img_Pilatus_{1}.tif'.format(data['scan'], int(x))) for x in imnums.ravel()]
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
					#         impath = os.path.join(imagedirectory, str(data['scan']), 'scan_{0}_img_Pilatus_{1}.tif'.format(data['scan'], int(imnums[m,n])))
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
