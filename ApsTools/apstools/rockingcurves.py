import numpy as np
from scipy.interpolate import interp2d
import cmocean
from matplotlib import patches as patches

def RockingCurve(ccds, qmat, reciprocal_ROI = [0, 0, -1, -1], real_ROI = [0, 0, -1, -1], plot = True, extent = None, min_counts = 50):
	"""
	Given Pilatus ccds, a realspace ROI, and reciprocal space ROI, qmat, fits a rocking curve

	ccds: [numimages x map_y x map_x x ccd_y x ccd_x] 
	"""

	### Set up our CCD arrays. 
	ccdsum = ccds.sum(0).sum(0).sum(0)	#summed ccd image over all points in rocking curve
	ROIccds = ccds[:,
		real_ROI[0]:real_ROI[2],real_ROI[1]:real_ROI[3],
		reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3]
		].astype(np.float32)	#all data used for rocking curve fit, trimmed to real and reciprocal ROIs
	mask = ROIccds.sum(4).sum(3).sum(0) <= min_counts	#any points on map without sufficient counts in reciprocal ROI are excluded from fit
	ROIccds[:,mask] = np.nanmean 
	sumROIccds = ROIccds.sum(axis = 0)	#total counts per realspace point over all unmasked realspace points

	### Initialize data vectors
	# hold q vector centroids from rocking curve fitting
	qxc = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	qyc = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	qzc = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	qmagc = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1])) #magnitude, used for d-spacing

	# hold q vector centroids, rotated into sample-normal coordinate system so qz//sample normal, qx/qy represent tilts
	qxc_r = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	qyc_r = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	qzc_r = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))

	# hold calculated tilt in x and y direction
	tiltx = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))
	tilty = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))

	# holds sample theta at which peak diffraction occurred. useful to see whether scanned sample thetas have found peak diffraction across realspace ROI
	peaksamth = np.zeros((sumROIccds.shape[0], sumROIccds.shape[1]))

	qxif = interp2d(list(range(ROI[3]-ROI[1])),list(range(ROI[2]-ROI[0])), qmat['qmat'][ROI[0]:ROI[2], ROI[1]:ROI[3], 0])
	qyif = interp2d(list(range(ROI[3]-ROI[1])),list(range(ROI[2]-ROI[0])), qmat['qmat'][ROI[0]:ROI[2], ROI[1]:ROI[3], 1])
	qzif = interp2d(list(range(ROI[3]-ROI[1])),list(range(ROI[2]-ROI[0])), qmat['qmat'][ROI[0]:ROI[2], ROI[1]:ROI[3], 2])
	qmagif = interp2d(list(range(ROI[3]-ROI[1])),list(range(ROI[2]-ROI[0])), qmat['qmat'][ROI[0]:ROI[2], ROI[1]:ROI[3], 3])

	def calc_centroid(im):
		"""
		given a 2-d array of values, returns tuple with array coordinates (m,n) of image centroid 
		"""
		vals = []
		for m,n in np.ndindex(im.shape[0], im.shape[1]):
			vals.append([m,n,im[m,n]])
		vals = np.array(vals)
		xc, yc = np.average(vals[:,:2], axis = 0, weights = vals[:,2])
		return (yc, xc)

	for m,n in np.ndindex(qmagc.shape[0], qmagc.shape[1]):
		try:
			[xc,yc] = calc_centroid(sumROIccds[m,n])
			qxc[m,n] = qxif(yc, xc)
			qyc[m,n] = qyif(yc, xc)
			qzc[m,n] = qzif(yc, xc)
			qmagc[m,n] = qmagif(yc,xc)
			tlist = ROIccds[:,m,n].sum(1).sum(1)
			tidx = np.where(tlist == np.nanmax(tlist))[0][0]
			peaksamth[m,n] = thvals[tidx]
		except:
			qxc[m,n] = np.nan
			qyc[m,n] = np.nan
			qzc[m,n] = np.nan
			qmagc[m,n] = np.nan
			peaksamth[m,n] = np.nan


	# calculate rotation matrix to move qmat coordinate system into diffraction plane normal coordinate system (using mean q vector as plane normal)
	u = []
	v = [0,0,1]	#adjust so mean q vector is parallel to [001] lattice vector
	for q_ in [qxc, qyc, qzc]:
		u.append(np.nanmean(q_))
	u = u / np.linalg.norm(u)
	rotation = calcRotation(u,v)

	# rotate all q centroids into plane normal coordinate system
	for m,n in np.ndindex(qxc.shape):
		for q_ in [qxc_r, qyc_r, qzc_r]:
			q_[m,n] = np.nan
		else:
			v1 = rotation@np.array([qxc[m,n], qyc[m,n], qzc[m,n]])
		for v_, q_ in zip(v1, [qxc_r, qyc_r, qzc_r]):
			q_[m,n] = v_

	#calculate tilt angles from qx/qy
	tiltx = 180*np.arccos(qxc_r/qmagc)/np.pi
	tilty = 180*np.arccos(qyc_r/qmagc)/np.pi
	# tiltx -= np.nanmean(tiltx)
	# tilty -= np.nanmean(tilty)

	dataout = {
		'q': np.array([qxc_r, qyc_r, qzc_r, qmagc]),
		'tiltx': tiltx,
		'tilty': tilty,
		'd': 2*np.pi/qmagc,
		'peaksamth': peaksamth
	}

	if plot:
		fig, ax = plt.subplots(2,2, figsize = (12,6))
		if extent is None:
			extent = [0, 0, ROIccds.shape[1], ROIccds.shape[2]]
		extent_area[1] *= -ROIccds.shape[1]/(ccds.shape[1]-1)
		extent_area[3] *= -ROIccds.shape[2]/(ccds.shape[2]-1)
		

		ax = np.transpose(ax)
		im = ax[0,1].imshow(qmagc, extent = extent_area, cmap = cmocean.cm.curl)# cmap = plt.cm.RdBu)
		yv, xv = np.meshgrid(
			np.linspace(extent_area[3], extent_area[2], qmagc.shape[0]),
			np.linspace(extent_area[0], extent_area[1], qmagc.shape[1]),
			sparse=False, indexing='ij'
		)
		ax[0,1].quiver(xv, yv, tilty.ravel(), tiltx.ravel(), angles = 'xy')#, width = 0.003)
		frgplt.Scalebar(ax[0,1], 1e-6, box_color = [0,0,0], box_alpha = 0.8, pad = 0.3)
		ax[0,1].set_xticks([])
		ax[0,1].set_yticks([])

		cb = plt.colorbar(im ,ax = ax[0,1], fraction = 0.036)
		# cb.set_label('$||Q||\ (A^{-1})$')
		cb.set_label('$d\ (A)$')
		ax[0,1].set_title('Rocking Curve Fitted')

		im = ax[0,0].imshow(peaksamth, cmap = plt.cm.RdBu, vmin = thvals[0], vmax = thvals[-1])
		cb = plt.colorbar(im, ax = ax[0,0], fraction = 0.036)
		cb.set_label('Peak Samth')
		ax[0,0].set_title('Rocking Curve Peak (white = good)')
		x = thvals
		y = []
		for rc in ROIccds:
			y.append(np.nanmean(rc))
		ax[1,0].plot(x,y,':o')
		ax[1,0].set_xlabel('Sample Theta')
		ax[1,0].set_ylabel('Average Counts')

		ax[1,1].imshow(ccdsum, norm=LogNorm(vmin=0.01, vmax=ccdsum.max()))
		rect = patches.Rectangle((ROI[1], ROI[0]),ROI[3]-ROI[1],ROI[2]-ROI[0],linewidth=1,edgecolor='r',facecolor='none')
		ax[1,1].add_artist(rect)
		ax[1,1].set_xticks([])
		ax[1,1].set_yticks([])

		plt.tight_layout()
		plt.show()

	

	return dataout