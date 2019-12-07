# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
sys.path.append('D:\APS Data\\2019c2_26\Data\Analysis')
from readMDA import readMDA
import cv2
import json
import h5py


mdafolder = 'D:\APS Data\\2019c2_26\Data\mda'
Image_folder = 'D:\APS Data\\2019c2_26\Data\Images'
h5_folder = 'D:\APS Data\\2019c2_26\Data\h5'

#%%

def readlogbook(scannum):
        logbookpath = 'D:\APS Data\\2019c2_26\Data\mda\logbook_verbose.txt'

        with open(logbookpath, 'r') as f:
                line = f.readline()
                start = 'SCAN #: {0:d}'.format(int(scannum))
                stop = '-----'
                found = False

                while line:
                        line = f.readline()
                        if start in line:
                                found = True
                                break
                if not found:
                        return False
                
                positioners = {
                        'datetime': f.readline()[:-1]   #get rid of the newline character
                }

                while line:
                        line = f.readline()
                        if stop in line:
                                break
                        pieces = line.split('\t')
                        positioners[pieces[0]] = float(pieces[1])
                        
        return positioners

def generate_energy_list(cal_offset = -0.0151744, cal_slope = 0.0103725, cal_quad = 0.00000):
        energy = [cal_offset + cal_slope*x + cal_quad*x*x for x in range(2048)]
        return energy

def Scan_Load(scannum, only3d = False, loadimages = False):     
        for f in os.listdir(mdafolder):
                if int(f.split('SOFT_')[1][:-4]) == scannum:
                        mdapath = os.path.join(mdafolder, f)
                        break


        if only3d:
                data = readMDA(mdapath, verbose=0, maxdim = 3)
                ndim = len(data)-1 if data[0]["dimensions"][-1] != 2048 else len(data)-2
        else:
                data = readMDA(mdapath, verbose=0, maxdim = 2)
        
        ndim = len(data)-1 if data[0]["dimensions"][-1] != 2048 else len(data)-2
        # scannum = mdapath.split(".")[0].split("_")[-1].lstrip("0")
        image_path = os.path.join(Image_folder, str(scannum))
        # os.chmod(image_path, 0o777)
        image_list = [imagefile.name for imagefile in os.scandir(image_path) if imagefile.name.endswith('.tif')]
        tif_index = np.array([int(filename.split(".")[0].split("_")[-1]) for filename in image_list])
        image_list = np.take(image_list , tif_index.argsort())
        tif_index.sort()
        nbin = 1
        mda_index = 0
        if ndim == 2:
                nfile = data[0]['dimensions'][0] * data[0]['dimensions'][1]
                for i in range(data[ndim].nd):
                        dname = data[ndim].d[i].name
                        if "FileNumber" in dname:
                                mda_index = np.array(data[2].d[i].data)
                                print (mda_index.max(), tif_index.max(), mda_index.min(), tif_index.min())
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
                'logbook': readlogbook(scannum),
                'image_list': image_list,
                'tif_index': tif_index,
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
#%%
def scantoh5(data, directory, loadimages = True):
        filepath = os.path.join(directory, 'scan_{0}.hdf5'.format(data['scan']))
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
                
                if loadimages:
                        numpts = 200
                        twothetaimage = dimages.create_dataset('twotheta', data = np.genfromtxt('D:\APS Data\\2019c2_26\Data\Analysis\qmat\\twotheta.csv', delimiter=','))
                        twothetaimage.attrs['description'] = 'Map correlating two-theta values to each pixel on diffraction ccd'
                        temp = dimages.create_dataset('gamma', data = np.genfromtxt('D:\APS Data\\2019c2_26\Data\Analysis\qmat\\gamma.csv', delimiter=','))
                        temp.attrs['description'] = 'Map correlating gamma values to each pixel on diffraction ccd'

                        tolerance = (twothetaimage[:].max()-twothetaimage[:].min()) / numpts
                        interp_twotheta = dpatterns.create_dataset('twotheta', data = np.linspace(twothetaimage[:].min()+tolerance/2, twothetaimage[:].max()-tolerance/2, numpts))
                        interp_twotheta.attrs['description'] = 'Twotheta values onto which ccd pixel intensities are collapsed.'
                        imnums = data['mda_index']-1      #files are saved offset by 1 for some reason
                        xrdcounts = dpatterns.create_dataset('cts', data = np.zeros((imnums.shape[0], imnums.shape[1], numpts)), chunks = True)
                        xrdcounts.attrs['description'] = 'Collapsed diffraction counts for each scan point.'
                        intxrdcounts = dpatterns.create_dataset('intcts', data = np.zeros((numpts,)))
                        intxrdcounts.attrs['description'] = 'Collapsed, area-integrated diffraction counts.'
                        images = None
                        for m, n in np.ndindex(imnums.shape):
                                impath = os.path.join(Image_folder, str(data['scan']), 'scan_{0}_img_Pilatus_{1}.tif'.format(data['scan'], int(imnums[m,n])))
                                im = cv2.imread(impath, -1)
                                if images is None:
                                        images = dimages.create_dataset('ccd', (imnums.shape[0], imnums.shape[1], im.shape[0], im.shape[1]), compression = "gzip", chunks = True)
                                        images.attrs['description'] = 'Raw ccd images for each scan point.'
                                        # images = [[None for n in range(imnums.shape[1])] for m in range(imnums.shape[0])]
                                images[m,n,:,:] = im
                                for tidx, tt in enumerate(interp_twotheta):
                                        xrdcounts[m,n,tidx] = np.sum(im[np.abs(twothetaimage[:]-tt) <= tolerance])
                                        intxrdcounts = intxrdcounts + xrdcounts[m,n,tidx]
                        
#%%
# sc = Scan_Load(mdas[187], only3d = True, loadimages = True)

# with open (os.path.join(h5_folder, 'temp.json'), 'w') as f:
#         json.dump(sc, f)


#%%

# scantoh5(sc, h5_folder)

#%%
