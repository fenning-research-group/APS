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
from scipy.interpolate import interp2d
import cmocean
from matplotlib import patches as patches
from frgtools import plotting as frgplt
import pickle
import multiprocessing.pool as mpp
from functools import partial
import skimage.filters as filters
from skimage.measure import label, regionprops
from PIL import Image
import pandas as pd
from .helpers import _load_image_rek
import re
from scipy.optimize import curve_fit

FRG_H5_PREFIX = '26idbSOFT_FRG_'

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

    return  (4*np.pi/wavelength)*np.sin(np.deg2rad(twotheta/2))

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

def twotheta_adjust(twotheta, energy, energy0 = None):
    '''
    converts twotheta value from initial x-ray energy (defaults to Cu-ka at 8.04 keV)
    to that at another x-ray energy

        twotheta: twotheta angle (degrees) at initial energy
        energy: energy to adjust angle to (keV)
        energy0: energy to adjust angle from (keV)
    '''

    if energy0 is None:
        print('No initial photon energy provided by user - assuming 8.040 keV (Cu-k-alpha)')
        energy0 = 8.040

    return q_to_twotheta(
                q = twotheta_to_q(twotheta, energy = energy0),
                energy = energy
                )


### scripts for rocking curve analysis

class RockingCurve:
    '''
    Performs 5d rocking curve processing of 2d maps of 2d diffraction ccds at various sample orientations 
    '''
    def __init__(self, ccds, samths, gamma, twotheta, energy, extent = None, correct_orientation = False):
        self.ccds = np.asarray(ccds) #5d array of rocking curve data [theta, realy, realx, recipy, recipx]
        bad_pixel_mask = self.ccds[0].sum(axis = (0,1)) < 0
        self.ccds[:,:,:,bad_pixel_mask] = 0
        if extent is None:
            self.extent = [0, self.ccds.shape[2], 0, self.ccds.shape[1]]
            self._add_scalebar = False
        else:
            self.extent = extent
            self._add_scalebar = True
            if correct_orientation:
                #assuming all rocking curves are taken in hybrid mode, orient the scan to match sample mounting orientation
                if self.extent[0] < self.extent[1]:
                    self.ccds = self.ccds[:,:,::-1]
                    self.extent[0], self.extent[1] = self.extent[1], self.extent[0]
                if self.extent[2] < self.extent[3]:
                    self.ccds = self.ccds[:,::-1]
                    self.extent[2], self.extent[3] = self.extent[3], self.extent[2]
        self.ccdsum_recip = ccds.sum(axis = (0,1,2))
        self.ccdsum_real = ccds.sum(axis = (0,3,4))

        self._samths = samths
        self._thetas = (np.pi/2 - np.deg2rad(np.asarray(samths)))[:, np.newaxis, np.newaxis]
        self._gamma = np.deg2rad(np.asarray(gamma))
        self._twotheta = np.deg2rad(np.asarray(twotheta))
        

        self._K = 2*np.pi/(12.398/energy) #convert beam energy to spatial frequency 

        # define q vector components per ccd pixel per sample orientation
        self.qy = self._K * np.sin(self._gamma) * np.ones(self._thetas.shape) # qy never changes, since sample rotates about qy direction. Multiply by thetas shape to copy qy for each samth
        self.qx = self._K * np.cos(self._gamma) * np.sin(self._twotheta) #calc for samth = 0, will adjust below
        self.qz = self._K * (np.cos(self._gamma)*np.cos(self._twotheta) - 1) #calc for samth = 0, will adjust below. subtract K so Z is defined as into the sample normal (ie [001] = sample normal, from front to back of sample)
        self.qmag = np.sqrt(self.qx**2 + self.qy**2 + self.qz**2)

        # account for rotation in theta (about qy) affecting qx, qz positions on detector
        self.qz, self.qx = self.qz*np.cos(self._thetas) - self.qx*np.sin(self._thetas), self.qz*np.sin(self._thetas) + self.qx*np.cos(self._thetas) #qz, qx changes as sample theta changes - rotation about qy
    
    def roi(self, reciprocal, real = None, threshold = 0, plot = True):
        '''
        define reciprocal and realspace rois for analysis. takes two lists of bounding coordinates.
            reciprocal = [ymin, ymax, xmin, xmax]
            real = [ymin, ymax, xmin, xmax]
        '''
        self.rec_i = slice(reciprocal[0], reciprocal[1])
        self.rec_j = slice(reciprocal[2], reciprocal[3])

        if real is None:
            self.real_i = slice(0, self.ccds.shape[1])
            self.real_j = slice(0, self.ccds.shape[2])
        else:
            self.real_i = slice(real[0], real[1])
            self.real_j = slice(real[2], real[3])

        self.roicts = self.ccds[:,:,:,self.rec_i, self.rec_j].sum(axis = (0,3,4))
        self.roicts[self.roicts < 0] = 0 #bugfix for negative value pixels, issue with ccd pixel damage
        self.mask = self.roicts >= threshold
                
        if plot:
            fig = plt.figure()
            gs = fig.add_gridspec(2,2)
            ax = []
            ax.append(fig.add_subplot(gs[1, 0]))
            ax.append(fig.add_subplot(gs[1, 1]))
            ax.append(fig.add_subplot(gs[0, :]))

            im = ax[0].imshow(self.roicts[self.real_i, self.real_j], origin = 'lower', cmap = plt.cm.gray, extent = self.extent, norm = LogNorm(vmin = np.max([1, self.roicts.min()]), vmax = self.roicts.max()))
            ax[0].imshow(self.apply_mask(self.roicts)[self.real_i, self.real_j], origin = 'lower', cmap = plt.cm.viridis, norm = LogNorm(vmin = np.max([1, self.roicts.min()]), vmax = self.roicts.max()), extent = self.extent, alpha = 0.6)
            ax[0].set_title('Realspace ROI')
            plt.colorbar(im, ax = ax[0], fraction = 0.046)

            if self._add_scalebar:
                frgplt.scalebar(1e-6, ax = ax[0]) #assume extent is given in microns
            im = ax[1].imshow(self.ccdsum_recip[self.rec_i, self.rec_j], norm = LogNorm(), extent = [reciprocal[i] for i in [2,3,1,0]])
            ax[1].set_aspect('equal')
            ax[1].set_title('Reciprocal ROI')
            plt.colorbar(im, ax = ax[1], fraction = 0.046)

            ax[2].imshow(self.ccdsum_recip, norm = LogNorm())
            rect = patches.Rectangle(
                (reciprocal[2], reciprocal[0]),
                reciprocal[3]-reciprocal[2],
                reciprocal[1]-reciprocal[0],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
                )
            ax[2].add_artist(rect)
            plt.tight_layout()
            plt.show()
    
    def apply_mask(self, im):
        im_ = im.copy().astype(float)
        im_[~self.mask] = np.nan
        return im_

    def fix_angles(self, angle):
        angle[angle>160] -= 180
        angle[angle<-160] += 180
        return angle

    def _fit_theta_fwhm(self, ccds):
        def gaussian(x, amplitude, center, sigma):
            return amplitude * np.exp(-(x-center)**2 / (2*sigma**2))
        x = np.array(self._samths)
        theta_fwhm = np.full((ccds.shape[1], ccds.shape[2]), np.nan)
        for m,n in tqdm(np.ndindex(theta_fwhm.shape), total = np.product(theta_fwhm.shape)):
            y = ccds[:,m,n].sum(axis = (1,2))
            p0 = [y.max(), x.mean(), 0.5]
            bounds = [[0,x.min(), 0], [y.max() * 2, x.max(), 10]]
    #         print(p0)
            try:
                popt, _ = curve_fit(gaussian, x, y, p0, bounds = bounds) 
                if popt[2] != 10: #max value, nonsensical. implies flat cts vs samth aka no counts.
                   theta_fwhm[m,n] = popt[2]
            except:
                pass
        return theta_fwhm

    def analyze(self, plot = True, stream = False):
        if self.rec_i is None:
            raise ValueError('Need to define the analysis regions of interest using ".roi()" before running rocking curve analysis')
        ccds_roi = self.ccds[:, self.real_i, self.real_j, self.rec_i, self.rec_j]
        ccds_roi[ccds_roi < 0] = 0 #buggy pixels on ccd can read negative. also possible from overflow if ccd files read in int8, other data type with too little range for data.
        q_mass = ccds_roi.sum(axis = (0,3,4)) #integrate diffraction counts over reciprocal roi + thetas for each realspace pixel
        self.qx_fitted = (ccds_roi * self.qx[:, np.newaxis, np.newaxis, self.rec_i, self.rec_j]).sum(axis = (0,3,4)) / q_mass #center of mass in qx per realspace pixel
        self.qy_fitted = (ccds_roi * self.qy[:, np.newaxis, np.newaxis, self.rec_i, self.rec_j]).sum(axis = (0,3,4)) / q_mass
        self.qz_fitted = (ccds_roi * self.qz[:, np.newaxis, np.newaxis, self.rec_i, self.rec_j]).sum(axis = (0,3,4)) / q_mass
        self.qmag_fitted = np.sqrt(self.qx_fitted**2 + self.qy_fitted**2 + self.qz_fitted**2)
        self.d_fitted = 2*np.pi/self.qmag_fitted

        self._align_q_to_sample_normal()
        self.theta_fwhm = self._fit_theta_fwhm(ccds_roi)

        self.yaw = np.rad2deg(np.arctan2(self.qx_fitted, self.qz_fitted)) #tilt in qxqz plane, degrees
        self.pitch = np.rad2deg(np.arctan2(self.qy_fitted, self.qz_fitted)) #tilt in qyqz plane, degrees
        self.roll = np.rad2deg(np.arctan2(self.qy_fitted, self.qx_fitted)) #tilt in qxqy plane, degrees
        
        self.yaw = self.fix_angles(self.yaw)
        self.pitch = self.fix_angles(self.pitch)
        self.roll = self.fix_angles(self.roll)
        if plot:
            plt.figure(figsize = (8,8))
            im = plt.imshow(self.apply_mask(self.d_fitted), cmap = cmocean.cm.curl, origin = 'lower', extent = self.extent)
            plt.colorbar(label = r'd-Spacing ($\AA$)')
            if self._add_scalebar:
                frgplt.scalebar(1e-6) #assume extent is given in microns

            xv, yv = np.meshgrid(np.linspace(self.extent[0], self.extent[1], self.yaw.shape[1]), np.linspace(self.extent[2], self.extent[3], self.yaw.shape[0]))
            if stream:
                xlim0 = plt.xlim()
                ylim0 = plt.ylim()
                mag = np.sqrt(self.yaw**2 + self.pitch**2)
                mag *= 4/np.nanmax(mag)
                plt.streamplot(
                    xv[0,:],
                    yv[:,0],
                    self.apply_mask(self.yaw),
                    self.apply_mask(self.pitch),
                    density = 1,
                    color = [0,0,0,0.7],
                    linewidth = mag
                    )
                plt.xlim(xlim0)
                plt.ylim(ylim0)
            else:
                plt.quiver(
                    xv[::2, ::2],
                    yv[::2, ::2],
                    self.apply_mask(self.yaw)[::2, ::2],
                    self.apply_mask(self.pitch)[::2, ::2],
                    angles = 'uv',
                    pivot = 'middle',
                    scale_units = 'xy',
                    headaxislength = 2,
                    headwidth = 6
                    )
            plt.show()

        return {'d': self.d_fitted, 'yaw': self.yaw, 'pitch': self.pitch}

    def _align_q_to_sample_normal(self):
        '''
        rotate qx, qy, qz such that the mean q vector is parallel to (001), 
        where (001) is normal to the diffracting planes. Internal method.
        '''
    
        qx0 = self.qx_fitted.mean()
        qy0 = self.qy_fitted.mean()
        qz0 = self.qz_fitted.mean()

        qx1 = 0
        qy1 = 0
        qz1 = 1

        if np.fabs(qx0) > 1e-2:
            self.__theta_xy = np.arctan2(-qy0, qx0) #(-qy0, qx0) for Laue
            self.qx_fitted, self.qy_fitted = self.qx_fitted*np.cos(self.__theta_xy) - self.qy_fitted*np.sin(self.__theta_xy), self.qx_fitted*np.sin(self.__theta_xy) + self.qy_fitted*np.cos(self.__theta_xy)
            qx0 = self.qx_fitted.mean()
        else:
            self.__theta_xy = 0
        
        self.__theta_xz = np.arctan2(-qx0, -qz0) #rotate diffraction vectors to be mean-centered in [001] direction, normal to diffracting planes in Bragg orientation. (-qx0, -qz0) will put this in Laue orientation.
        self.qz_fitted, self.qx_fitted = self.qz_fitted*np.cos(self.__theta_xz)+self.qx_fitted*np.sin(self.__theta_xz), -self.qz_fitted*np.sin(self.__theta_xz)+self.qx_fitted*np.cos(self.__theta_xz)
        # qz0 = self.qz_fitted.mean()

    def strain(self, d0 = None, plot = True):
        if d0 is None:
            d0 = self.d_fitted.mean()
            print('No reference d-spacing provided, all strain values are plotted relative to the mean d-spacing for now. This might not be accurate!')
        self.strain_fitted = (self.d_fitted/d0 - 1)*100 #strain in %

        if plot:
            plt.figure()
            plt.imshow(self.apply_mask(self.strain_fitted), cmap = cmocean.cm.curl, origin = 'lower', extent = self.extent)
            if self._add_scalebar:
                frgplt.scalebar(1e-6) #assume extent is given in microns

            cb = plt.colorbar()
            cb.set_label('Strain (%)')
            plt.show()

        return self.strain_fitted

    def export(self, threshold = 0):
        dataout = {
            'qx': self.qx_fitted,
            'qy': self.qy_fitted,
            'qz': self.qz_fitted,
            'qmag': self.qmag_fitted,
            'theta_fwhm': self.apply_mask(self.theta_fwhm),
            'samths': self._samths,
            'rotation': {'xy':self.__theta_xy, 'xz':self.__theta_xz},
            'counts': self.roicts,
            'tiltx': self.apply_mask(self.pitch),
            'tilty': self.apply_mask(self.yaw),
            'd': self.apply_mask(self.d_fitted),
            'mask': self.mask,
            'extent': self.extent,
            'geometry': str.lower('Bragg'),
            'summedccd': self.ccdsum_recip,
            'recip_roi': [self.rec_i, self.rec_j]
        }

        return dataout

def rocking_curve(ccds, qmat, thvals, reciprocal_ROI = [0, 0, None, None], real_ROI = [0, 0, None, None], plot = True, extent = None, min_counts = 50, stream = False, savepath = None, geometry = 'bragg'):
    """
    Given Pilatus ccds, a realspace ROI, and reciprocal space ROI, qmat, fits a rocking curve

    ccds: 5d numpy array with all ccd images. [numimages, map_y, map_x, ccd_y, ccd_x] 
    qmat: Pilatus calibration dictionary generated by MATLAB script. Converts detector pixels to reciprocal space coordinates
    real_ROI: bounds in realspace map coordinates to process rocking curve. This should isolate the sample region of interest [ymin, xmin, ymax, xmax]
    reciprocal_ROI: bounds in detector CCD to process rocking curve. This should isolate the diffraction spot of interest [ymin, xmin, ymax, xmax]
    min_counts: threshold counts. Pixels with total counts below this value across the sum of all maps are ignored as nan.
    geometry: enter 'bragg' or 'laue', affects the tilt calculation

    plot: boolean flag to determine whether plots are generated
    savepath: filepath to save plot images to
    """

    ### Set up our CCD arrays. 
    ccdsum = ccds.sum(0).sum(0).sum(0)  #summed ccd image over all points in rocking curve
    ROIccds = ccds[:,
        real_ROI[0]:real_ROI[2],real_ROI[1]:real_ROI[3],
        reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3]
        ].astype(np.float32)    #all data used for rocking curve fit, trimmed to real and reciprocal ROIs
    mask = ROIccds.sum(4).sum(3).sum(0) <= min_counts   #any points on map without sufficient counts in reciprocal ROI are excluded from fit
    ROIccds[:,mask] = np.nan 
    sumROIccds = ROIccds.sum(axis = 0)  #total counts per realspace point over all unmasked realspace points

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

    qxif = interp2d(list(range(reciprocal_ROI[3]-reciprocal_ROI[1])),list(range(reciprocal_ROI[2]-reciprocal_ROI[0])), qmat['qmat'][reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3], 0])
    qyif = interp2d(list(range(reciprocal_ROI[3]-reciprocal_ROI[1])),list(range(reciprocal_ROI[2]-reciprocal_ROI[0])), qmat['qmat'][reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3], 1])
    qzif = interp2d(list(range(reciprocal_ROI[3]-reciprocal_ROI[1])),list(range(reciprocal_ROI[2]-reciprocal_ROI[0])), qmat['qmat'][reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3], 2])
    qmagif = interp2d(list(range(reciprocal_ROI[3]-reciprocal_ROI[1])),list(range(reciprocal_ROI[2]-reciprocal_ROI[0])), qmat['qmat'][reciprocal_ROI[0]:reciprocal_ROI[2], reciprocal_ROI[1]:reciprocal_ROI[3], 3])

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
    def calcRotation(u,v):
        cos = np.dot(u,v)
        cross = np.cross(u,v)
        sin = np.linalg.norm(cross)
        skew = np.array([
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0]
        ])
        
        rotation = np.eye(3) + skew + np.dot(skew, skew)*(1-cos)/(sin**2)
        return rotation


    u = []
    if str.lower(geometry) == 'bragg':
        v = [0,0,1] #adjust so mean q vector is parallel to [001] lattice vector (assuming Bragg!)
    elif str.lower(geometry) == 'laue':
        v = [0,0,-1]
    else:
        print('Invalid geometry provided - must be \'bragg\' or \'laue\'. Assuming bragg for now, tilt values may be incorrect.')
        v = [0,0,1]

    for q_ in [qxc, qyc, qzc]:
        u.append(np.nanmean(q_))
    u = u / np.linalg.norm(u)
    rotation = calcRotation(u,v)

    # rotate all q centroids into plane normal coordinate system
    for m,n in np.ndindex(qxc.shape):
        if np.isnan(qxc[m,n]):
            for q_ in [qxc_r, qyc_r, qzc_r]:
                q_[m,n] = np.nan
        else:
            v1 = rotation@np.array([qxc[m,n], qyc[m,n], qzc[m,n]])
            for v_, q_ in zip(v1, [qxc_r, qyc_r, qzc_r]):
                q_[m,n] = v_

    #calculate tilt angles from qx/qy
    tiltx = 180*np.arcsin(qxc_r/qzc_r)/np.pi
    tilty = 180*np.arcsin(qyc_r/qzc_r)/np.pi
    tiltx -= np.nanmean(tiltx)
    tilty -= np.nanmean(tilty)


    dataout = {
        'q': np.array([qxc_r, qyc_r, qzc_r, qmagc]),
        'q_raw': np.array([qxc, qyc, qzc, qmagc]),
        'rotation': rotation,
        'counts': sumROIccds.sum(axis = (2,3)),
        'tiltx': tiltx,
        'tilty': tilty,
        'd': 2*np.pi/qmagc,
        'peaksamth': peaksamth,
        'extent': extent,
        'geometry': str.lower(geometry),
        'summedccd': ccdsum,
        'ccdroi': reciprocal_ROI
    }

    if plot is True or savepath is not None:
        if extent is None:
            extent = [0, ROIccds.shape[2], 0,ROIccds.shape[1]]
        #extent[1] *= -ROIccds.shape[1]/(ccds.shape[1]-1)
        #extent[3] *= -ROIccds.shape[2]/(ccds.shape[2]-1)
        
        # fig, ax = plt.subplots(2,2, figsize = (12,6))
        # ax = np.transpose(ax)

        fig = plt.figure(figsize = (14,6))
        gs = fig.add_gridspec(2,4)
        ax = [[None, None], [None, None]]
        ax[0][0] = fig.add_subplot(gs[0, 0])        #peak samth
        ax[0][1] = fig.add_subplot(gs[:, 2:])   #d spacing
        ax[1][0] = fig.add_subplot(gs[0,1])     #average counts per samth
        ax[1][1] = fig.add_subplot(gs[1, 0:2])   #ROI on Pilatus



        im = ax[0][1].imshow(dataout['d'], extent = extent, cmap = cmocean.cm.curl, origin = 'lower')# cmap = plt.cm.RdBu)
        xv, yv = np.meshgrid(np.linspace(extent[0], extent[1], tiltx.shape[1]), np.linspace(extent[2], extent[3], tiltx.shape[0]))
        if stream:
            xlim0 = ax[0][1].get_xlim()
            ylim0 = ax[0][1].get_ylim()
            mag = np.sqrt(tiltx**2 + tilty**2)
            mag *= 4/np.nanmax(mag)
            ax[0][1].streamplot(
                xv[0,:],
                yv[:,0],
                tiltx,
                tilty,
                density = 1,
                color = [0,0,0,0.7],
                linewidth = mag
                )
            ax[0][1].set_xlim(xlim0)
            ax[0][1].set_ylim(ylim0)
        else:
            ax[0][1].quiver(
                xv[::2, ::2],
                yv[::2, ::2],
                tiltx[::2, ::2],
                tilty[::2, ::2],
                angles = 'uv',
                pivot = 'middle',
                scale_units = 'xy',
                headaxislength = 2,
                headwidth = 6
                )
        frgplt.scalebar(1e-6, ax = ax[0][1], box_color = [0,0,0], box_alpha = 0.8, pad = 0.3)
        ax[0][1].set_xticks([])
        ax[0][1].set_yticks([])

        cb = plt.colorbar(im ,ax = ax[0][1], fraction = 0.036)
        # cb.set_label('$||Q||\ ({\AA}^{-1})$')
        cb.set_label('$d\ (\AA)$')
        ax[0][1].set_title('Rocking Curve Fitted')

        im = ax[0][0].imshow(peaksamth, cmap = plt.cm.coolwarm, vmin = thvals[0], vmax = thvals[-1], origin = 'lower')
        cb = plt.colorbar(im, ax = ax[0][0], fraction = 0.036)
        cb.set_label('Peak Samth')
        ax[0][0].set_title('Rocking Curve Peak (Gray = Good)')
        x = thvals
        y = []
        for rc in ROIccds:
            y.append(np.nanmean(rc))
        ax[1][0].plot(x,y,':o')
        ax[1][0].set_xlabel('Sample Theta')
        ax[1][0].set_ylabel('Average Counts')
        ax[1][0].set_title('Map-Averaged Diffraction Counts in CCD ROI')

        ax[1][1].imshow(ccdsum, norm=LogNorm(vmin=0.01, vmax=ccdsum.max()), cmap = plt.cm.gray)
        rect = patches.Rectangle((reciprocal_ROI[1], reciprocal_ROI[0]),reciprocal_ROI[3]-reciprocal_ROI[1],reciprocal_ROI[2]-reciprocal_ROI[0],linewidth=1,edgecolor='r',facecolor='none')
        ax[1][1].add_artist(rect)
        ax[1][1].set_xticks([])
        ax[1][1].set_yticks([])
        ax[1][1].set_title('Diffraction CCD ROI')

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, bbox_inches = 'tight', dpi = 300)
        if plot:
            plt.show()
        else:
            plt.close()

    return dataout

def plot_rockingcurve(d, tiltx, tilty, stream = False, extent = None, ax = None):
    if ax is None:
        ax = plt.gca()
    if extent is None:
        extent = [0, d.shape[1], 0, d.shape[0]]

    im = ax.imshow(d, cmap = cmocean.cm.curl, origin = 'lower', extent = extent)
    # plt.colorbar(label = r'd-Spacing ($\AA$)')

    xv, yv = np.meshgrid(np.linspace(extent[0], extent[1], tiltx.shape[1]), np.linspace(extent[2], extent[3], tilty.shape[0]))
    if stream:
        xlim0 = plt.xlim()
        ylim0 = plt.ylim()
        mag = np.sqrt(tiltx**2 + tilty**2)
        mag *= 4/np.nanmax(mag)
        ax.streamplot(
            xv[0,:],
            yv[:,0],
            tiltx,
            tilty,
            density = 1,
            color = [0,0,0,0.7],
            linewidth = mag
            )
        ax.set_xlim(xlim0)
        ax.set_ylim(ylim0)
    else:
        ax.quiver(
            xv[::2, ::2],
            yv[::2, ::2],
            tiltx,
            tilty,
            angles = 'uv',
            pivot = 'middle',
            scale_units = 'xy',
            headaxislength = 2,
            headwidth = 6
            )

### plotting functions
def overlay_ccd_angles(calibration_image, levels, label_levels = None, ax = None, label_kwargs = {}, contour_kwargs = {}):
    '''
    plots contours over a ccd, especially useful for the Pilatus detector. Example:

        plt.imshow(pilatus_ccd_image) #draw measured ccd image
        overlay_ccd_angles(qmat['twotheta'], [20,30,40,50]) #draw twotheta contours every 10 degrees
        overlay_ccd_angles(qmat['gamma'], [0,5,10]) #draw gamma contours every 5 degrees
        overlay_ccd_angles(qmat['twotheta'], [20,39], label_levels = dict(20 = '002', 39 = '004'), contour_kwargs = dict(colors = plt.cm.tab10(1), linestyles = 'dashed')) #draws expected reflections in dashed orange lines, and labels by reflection rather than degrees
        
    '''
    _contour_kwargs = dict(
        levels = levels,
        colors = 'w',
        alpha = 0.6,
        linestyles = 'solid'
        )
    for k,v in contour_kwargs.items():
        _contour_kwargs[k] = v


    if label_levels is None:
        label_levels = levels
    elif type(label_levels) is dict:
        label_kwargs['fmt'] = label_levels
        label_levels = list(label_levels.keys())
    _label_kwargs = dict(
        levels = label_levels,
        inline = True,
        inline_spacing = 10,
        )
    for k,v in label_kwargs.items():
        _label_kwargs[k] = v


    if ax is None:
        ax = plt.gca()

    X = np.arange(calibration_image.shape[1])
    Y = np.arange(calibration_image.shape[0])
    cs = ax.contour(X, Y, calibration_image, **_contour_kwargs)
    ax.clabel(cs, **_label_kwargs)

### scripts for working with Daemon-generated H5 Files

def find_ROIs(ccds, thvals, bin_size = 1, min_intensity = 3, centroid_distance_threshold = 10, area_threshold = 10, filter_tolerance = 2, plot = True, log = False, n_processes=1):
    def updateRegions(r, regions, centroid_distance_threshold = centroid_distance_threshold):
        for ridx, r_ in enumerate(regions):
            if np.linalg.norm([a - b for a,b in zip(r.centroid, r_.centroid)]) <= centroid_distance_threshold: #euclidean dsitance - centroids are close enough that we are assuming these regions are the same
                if r.mean_intensity > r_.mean_intensity:   #the diffraction at this point is a better representation of diffraction region on ccd
                    regions[ridx] = r
                return regions
        #if we make it here, we have a new region
        regions.append(r)
        return regions

    ccds = np.array(ccds)
    iterable = []
    m = 0
    while m <= ccds.shape[1] - bin_size:
        n = 0
        while n <= ccds.shape[2] - bin_size:
            iterable.append((m,n))
            n = n + bin_size
        m = m + bin_size
    if n_processes==1:
        # iterable = [(x[0], x[1]) for x in np.ndindex(ccds.shape[1]-bin_size, ccds.shape[2]-bin_size)]
        # iterable = [(x[0], x[1]) for x in np.ndindex(2,2)]
        if log:
            allregions = [_findRegionsLi(
                            *r,
                            ccds = np.log(ccds.sum(0)),
                            min_area = area_threshold,
                            bin_size = bin_size,
                            min_intensity = min_intensity,
                            tolerance = filter_tolerance)
                            for r in tqdm(iterable)
                            ]
        else:
            allregions = [_findRegionsLi(
                            *r,
                            ccds = ccds.sum(0),
                            min_area = area_threshold,
                            bin_size = bin_size,
                            min_intensity = min_intensity,
                            tolerance = filter_tolerance)
                            for r in tqdm(iterable)
                            ]            
    else:
        print('Starting multiprocessing pool')
        allregions = []
        os.environ["OPENBLAS_MAIN_FREE"] = "1"
        with mp.Pool(n_processes) as p:
            print('Segmenting regions per map point')
            # iterable = [(x[0], x[1]) for x in np.ndindex(ccds.shape[1]-bin_size, ccds.shape[2]-bin_size)]
            # iterable = [(x[0], x[1]) for x in np.ndindex(2,2)]
            if log:
                for r in tqdm(p.istarmap(partial(_findRegionsLi, ccds = np.log(ccds.sum(0)), min_area = area_threshold, bin_size = bin_size, min_intensity = min_intensity, tolerance = filter_tolerance), iterable, chunksize = 150), total=len(iterable)):
                    # if type(r) is not list:
                    #   r = [r]
                    #   for r_ in r:
                    #       if r_.area >= 10:
                    #           updateRegions(r_, regions)
                    allregions.append(r)
            else:
                for r in tqdm(p.istarmap(partial(_findRegionsLi, ccds = ccds.sum(0), min_area = area_threshold, bin_size = bin_size, min_intensity = min_intensity, tolerance = filter_tolerance), iterable, chunksize = 150), total=len(iterable)):
                    # if type(r) is not list:
                    #   r = [r]
                    #   for r_ in r:
                    #       if r_.area >= 10:
                    #           updateRegions(r_, regions)
                    allregions.append(r)

    regions = []
    for rs in tqdm(allregions):
        if type(rs) is not list:
            rs = [rs]
        for r in rs:
            if r.area >= area_threshold:
                updateRegions(r, regions)

    if plot:
        roiimg = np.full((ccds.shape[3], ccds.shape[4]), np.nan)
        bgimg = np.ones(roiimg.shape)
        for ridx, r in enumerate(regions):
            for m, n in r.coords:
                roiimg[m,n] = ridx

        plt.figure()
        plt.imshow(bgimg)
        plt.imshow(roiimg, cmap = plt.cm.nipy_spectral)
        plt.title('{} Regions Identified'.format(len(regions)))
        plt.show()

    columns = [x for x in dir(regions[0]) if not x.startswith('_')]
    dfData = {c:[] for c in columns}
    for r in regions:
        for c in columns:
            dfData[c].append(getattr(r,c))
    df = pd.DataFrame(dfData)
    
    summedccd = ccds.sum(axis = (0,1,2))
    def summedintensity(bbox):
        cts = summedccd[
            bbox[0]:bbox[2],
            bbox[1]:bbox[3]
        ].sum()
        return cts
    df['summed_max_intensity'] = df['bbox'].apply(summedintensity)  

    def meanintensity(bbox):
        cts = summedccd[
            bbox[0]:bbox[2],
            bbox[1]:bbox[3]
        ].mean()
        return cts
    df['summed_mean_intensity'] = df['bbox'].apply(summedintensity) 

    df.sort_values(['summed_max_intensity','summed_mean_intensity', 'max_intensity', 'area'], inplace = True, ascending = False)

    return df

def diffraction_map(fpath, twotheta = None, q = None, ax = None, tol = 2):
    """
    Plots maps of diffraction intensty across map area, given the following:

        fpath: filepath to daemon-generated H5 file
        twotheta: 
                    One of the following must be provided. Each would be a list of up to 5 values.
        q: 
        ax: The matplotlib axis to display to. if none is provided, a new one will be generated
        tol: the tolerance/window size of diffraction signal to count intensity for. (counts intensity at +/- tol). defaults to 2

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

### Helper object to work with raw scan files.

class RawDataHelper():
    def __init__(self, rootdirectory, xrdlib = []):
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
        #   self.qmat = json.load(f)
        self.imageDirectory = os.path.join(self.rootDirectory, 'Images')

        self.xrdlib = xrdlib

    # Plotting functions

    def twotheta_waterfall(self, scannum, numtt = 200, timestep = 1, xrdlib = None, hotccdthreshold = np.inf, ax = None):
        plotAtTheEnd = False
        if ax is None:
            fig, ax = plt.subplots(figsize = (8, 4))
            plotAtTheEnd = True

        if xrdlib is None:
            xrdlib = self.xrdlib

        with open(os.path.join(self.qmatDirectory, 'qmat.json'), 'r') as f:
            qmat = json.load(f)

        imdir = os.path.join(self.imageDirectory, str(scannum))
        imfids = [os.path.join(imdir, x) for x in os.listdir(imdir) if 'Pilatus' in x]  #currently only anticipates Pilatus CCD images

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

    def sum_CCD(self, scannum, numtt = 200, xrdlib = None, hotccdthreshold = np.inf, ax = None):
        plotAtTheEnd = False
        if ax is None:
            fig, ax = plt.subplots(figsize = (8, 4))
            plotAtTheEnd = True

        if xrdlib is None:
            xrdlib = self.xrdlib
            
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
                # cmask = cmask.astype(float)
                # cmask[cmask == 0] = np.nan
                # mask = np.array([mask*c_ for c_ in c]).reshape(195, 487, 4)
                # return mask
                # mask[mask == 0] = np.nan
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
        #   self.qmat = json.load(f)
        self.imageDirectory = os.path.join(self.rootDirectory, 'Images')

        #self.Listener() #start the daemon

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
        while True: #keep running unless manually quit by ctrl-c
            mostRecentScan = findMostRecentScan()   #get most recent scan number
            if self.lastProcessedScan < findMostRecentScan: #have we looked at this scan yet?
                scanFunction = lookupScanFunction(mostRecentScan)   #if not, lets see if its a scan we want to convert to an h5 file
                if scanFunction in functions:   #if it is one of our target scan types (currently only works on scan2d as of 20191206)
                    if epics.caget("26idc:filter:Fi1:Set") == 0:    #make sure that the scan has completed (currently using filter one being closed as indicator of completed scan)
                        try:
                            self.MDAToH5(scannum = mostRecentScan)  #if we passed all of that, fit the dataset
                        except:
                            print('  Error converting scan {} to H5'.format(mostRecentScan))
                        self.lastProcessedScan = mostRecentScan
                else:
                    self.lastProcessedScan = mostRecentScan     #if the scan isnt a fittable type, set the scan number so we dont look at it again

            time.sleep(5)   # check for new files every 5 seconds



### Helper Functions

def _findRegionsFlatThreshold(m,n, ccds, min_area = 2, min_intensity = 0.5, bin_size = 5):

    ccds0 = np.log(ccds[m:m+bin_size,n:n+bin_size].sum(0).sum(0))
    ccds0[np.isnan(ccds0)] = 0
    ccds0[np.abs(ccds0) == np.inf] = 0
    # ccd_thresh = filters.threshold_li(ccds0,tolerance = 0.45) 
    # ccd_thresh = filters.threshold_local(ccds0,block_size=41, offset=0) 
    mask = ccds0 > 1.5
    mask_labels = label(mask)
    # return regionprops(mask_labels, intensity_image = ccds0)
    return [x for x in regionprops(mask_labels, intensity_image = ccds0) if x.area >= min_area and x.max_intensity > min_intensity] #, intensity_image = ccds0)

def _findRegionsLi(m,n, ccds, min_area = 10, min_intensity = 3, bin_size = 2, tolerance = 2):

    ccds0 = np.log(ccds[m:m+bin_size,n:n+bin_size].sum(0).sum(0))
    ccd_thresh = filters.threshold_li(ccds0, tolerance = tolerance) 
    # ccd_thresh = filters.threshold_local(ccds0,block_size=41, offset=0) 
    mask = ccds0 > ccd_thresh
    mask_labels = label(mask)
    # return regionprops(mask_labels, intensity_image = ccds0)
    return [x for x in regionprops(mask_labels, intensity_image = ccds0) if x.area >= min_area and x.max_intensity > min_intensity] #, intensity_image = ccds0)


def __istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap, Python 3.8+
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = __istarmap

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
    filepath = os.path.join(h5directory, '{0}{1:04d}.h5'.format(FRG_H5_PREFIX, data['scan']))
    with h5py.File(filepath, 'w') as f:
            
        info = f.create_group('/info')
        info.attrs['description'] = 'Metadata describing scan parameters, sample, datetime, etc.'
        temp = info.create_dataset('scan', data = data['scan'])
        temp.attrs['description'] = 'Scan number'
        temp = info.create_dataset('ndim', data = data['ndim'])
        temp.attrs['description'] = 'Number of dimensions in scan dataset'
        temp = info.create_dataset('x', data = data['positioners']['values'][1])
        temp.attrs['description'] = 'inner loop coordinates'
        temp.attrs['name'] = data['positioners']['names'][1]
        temp = info.create_dataset('y', data = data['positioners']['values'][0])
        temp.attrs['description'] = 'outer loop coordinates'
        temp.attrs['name'] = data['positioners']['names'][0]

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
        
        f.flush()   # write xrf data to disk

        if loadimages:
                numpts = 200
                twothetaimage = dimages.create_dataset('twotheta', data = np.genfromtxt(twothetaccdpath, delimiter=','))
                twothetaimage.attrs['description'] = 'Map correlating two-theta values to each pixel on diffraction ccd'
                temp = dimages.create_dataset('gamma', data = np.genfromtxt(gammaccdpath, delimiter=','))
                temp.attrs['description'] = 'Map correlating gamma values to each pixel on diffraction ccd'
                tolerance = (twothetaimage[:].max()-twothetaimage[:].min()) / numpts
                interp_twotheta = dpatterns.create_dataset('twotheta', data = np.linspace(twothetaimage[:].min(), twothetaimage[:].max(), numpts))
                interp_twotheta.attrs['description'] = 'Twotheta values onto which ccd pixel intensities are collapsed.'
                imnums = data['mda_index']-1      #files are saved offset by 1 for some reason
                xrdcounts = dpatterns.create_dataset('cts', data = np.zeros((imnums.shape[0], imnums.shape[1], numpts)), chunks = True)
                xrdcounts.attrs['description'] = 'Collapsed diffraction counts for each scan point.'
                intxrdcounts = dpatterns.create_dataset('intcts', data = np.zeros((numpts,)))
                intxrdcounts.attrs['description'] = 'Collapsed, area-integrated diffraction counts.'

                allimgpaths = [os.path.join(imagedirectory, str(data['scan']), f) for f in os.listdir(os.path.join(imagedirectory, str(data['scan'])))]
                allimgpathnums = [int(re.match(r'scan_\d*_\D*_(\d*).tif', os.path.basename(f)).group(1)) for f in allimgpaths]
                imgpaths = [allimgpaths[allimgpathnums.index(pn)] for pn in imnums.ravel()]
                # imgpaths = [os.path.join(imagedirectory, str(data['scan']), 'scan_{0}_pil_{1:05d}.tif'.format(data['scan'], int(x))) for x in imnums.ravel()]
                print('Loading Images')
                # print(imgpaths)
                # imgdata = p.starmap(cv2.imread, [(x, -1) for x in imgpaths])
                imgdata = p.starmap(_load_image_rek, [(x,) for x in imgpaths])
                d = imgdata[0].shape
                imgdata = np.array(imgdata).reshape(imnums.shape[0], imnums.shape[1], d[0], d[1])

                images = dimages.create_dataset('ccd', data = imgdata, compression = 'gzip', chunks = True)
                print('Fitting twotheta')

                f.flush()
                ttmask = [np.abs(twothetaimage - tt_) <= tolerance for tt_ in interp_twotheta]

                for m,n in tqdm(np.ndindex(imnums.shape), total = imnums.shape[0] * imnums.shape[1]):
                    # for tidx, tt in enumerate(interp_twotheta):
                        # im = imgdata[m,n]
                        # xrdcounts[m,n,tidx] = np.sum(im[np.abs(twothetaimage[:]-tt) <= tolerance])    #add diffraction from all points where twotheta falls within tolerance
                    xrdcounts[m,n] = p.starmap(np.sum, [(imgdata[m,n][ttmask_],) for ttmask_ in ttmask])
                intxrdcounts[()] = xrdcounts[()].sum(axis=0).sum(axis=0)    #sum across map dimensions

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
