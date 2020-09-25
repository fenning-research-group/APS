import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
import matplotlib.transforms as transforms

packageDir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(packageDir, 'include', 'xrfEmissionLines.json'), 'r') as f:
	emissionLines = json.load(f)

### Plotting Functions

def plot_XRF_lines(elements, ax = None, maxlinesperelement = np.inf, tickloc = 'bottom', tickstagger = 0, ticklength = 0.05):     
    '''
    Given a list of elements, adds tick marks on a matplotlib plot axis indicated xrf emission lines for those elements.
    If ax is not specified, uses most recent plt axis.
    '''   
    if ax is None:
    	ax = plt.gca()
    
    xlim0 = ax.get_xlim()
    stagger = 0
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for idx, element in enumerate(elements):
        color = plt.cm.get_cmap('tab10')(idx)
        if tickloc == 'bottom':
            texty = 0.01 + 0.05*idx
        else:
            texty = 0.99 - 0.05*idx
        ax.text(0.99, texty, element,
            fontname = 'Verdana', 
            fontsize = 12,
            fontweight = 'bold',
            color = color, 
            transform = ax.transAxes,
            horizontalalignment = 'right',
            verticalalignment = tickloc)

        plotted = 0
        for line in emissionLines[element]['xrfEmissionLines']:
            if (line <= xlim0[1]) and (line >= xlim0[0]):
                # plt.plot([line, line], [0.98 - (idx+1)*ticklength, 0.98 - idx*ticklength], transform = trans, color = color, linewidth = 1.5)
                if tickloc == 'bottom':
                	plt.plot([line, line], [0.01 + stagger, 0.01 + ticklength + stagger], transform = trans, color = color, linewidth = 1.5)
                else:
                	plt.plot([line, line], [0.99 - stagger, 0.99 - ticklength - stagger], transform = trans, color = color, linewidth = 1.5)
                plotted += 1
            if plotted >= maxlinesperelement:
                break
        stagger += tickstagger

### Functions to return material properties ###

def scattering_factor(element, energy):
    '''
    returns the real and imaginary x ray scattering factors for an element at a given energy (keV).
    '''

    dataDir = os.path.join(packageDir, 'include', 'scatfacts')
    # dataElements = [x[:-4] for x in os.listdir(dataDir)]

    fid = os.path.join(dataDir, '{0}.nff'.format(str.lower(element)))
    with open(fid, 'r') as f:
        data = np.genfromtxt(f)[1:, :]
    e_raw = data[:, 0]/1000 #convert eV to keV
    f1_raw = data[:, 1]
    f2_raw = data[:, 2]

    f1 = np.interp(energy, e_raw, f1_raw)
    f2 = np.interp(energy, e_raw, f2_raw)
    return f1, f2

def molar_mass(element):
    '''
    looks up molar mass of a given atom
    '''
    dataPath = os.path.join(packageDir, 'include', 'Molar Masses.txt')
    molar_mass = None
    with open(dataPath, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] == element:
                molar_mass = float(row[1])
                break
    return molar_mass

class Material:
    '''
    Class that, for a defined material, can generate x-ray attenuation coefficients
    and related values.
    '''
    def __init__(self, elements, density):
        '''
        elements: dictionary of elements and their molar fraction of the material. 
                            ie, for FeCu2: {'Fe':1, 'Cu':2} 
        density: overall density of material (g/cm3)
        '''
        self.elements = elements
        self.density = density

    def attenuation_coefficient(self, energy):
        '''
        returns x-ray attenuation coefficient, in cm-1, given:
            energy: x-ray energy(ies) (keV)
        '''
        N_a = 6.022e23
        c = (1e-19)/(np.pi*0.9111)  # keV*cm^2
        energy = np.array(energy)
        f2 = 0
        mass = 0
        for i, (el, num) in self.elements.items():
            _, f2temp = scattering_factor(el, energy)
            f2 = f2 + num*f2temp
            mass = mass + num*molar_mass(el)

        mu = (self.density*N_a/mass) * (2*c/energy) * f2
        return mu
    
    def attenuation_length(self, energy):
        '''
        returns x-ray attenuation length (distance for transmitted intensity to
        decay to 1/e), in cm
        '''
        mu = self.attenuation_coefficient(elements, numElements, density, energy)
        return 1/mu

    def transmission(self, thickness, energy):
        '''
        returns fraction of x-ray intensity transmitted through a sample, defined by
            thickness: path length of x rays (cm)
            energy: x-ray energy (keV)
        '''

        mu = attenuation_coefficient(energy)
        t = np.exp(-mu*thickness)

        return t

### self-absorption

def self_absorption_film(material: Material, thickness, incidentenergy, xrfenergy, sampletheta, detectortheta):
    '''
    returns fraction of incident beam power that is causes fluorescence, transmits through a sample, and reaches the XRF detector. 
    This calculation assumes no secondary fluorescence/photon recycling. The returned fraction is the apparent signal after 
    incident beam attenuation and exit fluorescence attenuation - dividing the measured XRF value by this fraction should
    approximately correct for self-absorption losses and allow better comparison of fluorescence signals in different energy
    ranges.

    Calculations are defined by:

        material: xrf.Material class 
        thickness: Sample thickness - NOT PATH LENGTH (cm)
        incidentenergy: x-ray energy (keV) of incoming beam
        xrfenergy: x-ray energy (keV) of XRF signal
        sampletheta: angle (degrees) between incident beam and sample normal
        detectortheta: angle    (degrees) between incident beam and XRF detector axis
    '''

    incidentAttCoeff = material.attenuation_coefficient(incidentenergy)
    exitAttCoeff = material.attenuation_coefficient(xrfenergy)
    
    incident_theta = np.deg2rad(sampletheta)
    exit_theta = np.deg2rad(detectortheta - sampletheta)

    c = np.abs(incidentAttCoeff/np.cos(incident_theta)) + np.abs(exitAttCoeff/np.cos(exit_theta))

    xrfFraction = (1/thickness) * (1/c) * (1 - np.exp(-c*thickness))

    return xrfFraction

# def self_absorption_particle(elements, numElements, density, thickness, xscale, yscale, zscale, zstep = 5, incidentenergy, xrfenergy, sampletheta):
#     '''
#     returns fraction of x-ray fluorescence excited, transmitted through a sample, and reaching to an XRF detector. This
#     calculation assumes no secondary fluorescence/photon recycling. The returned fraction is the relative apparent signal after 
#     incident beam attenuation and exit fluorescence attenuation - dividing the measured XRF value by this fraction should
#     approximately correct for self-absorption losses and allow better comparison of fluorescence signals in different energy
#     ranges. This code assumes the 

#     Calculations are defined by:

#         elements: list of elements ['Fe', 'Cu']
#         numElements: list of numbers corresponding to elemental composition. [1,2] for FeCu2
#         density: overall density of material (g/cm3)
#         thickness: 2d array of sample thickness - NOT PATH LENGTH (cm)
#         xscale: cm per pixel
#         yscale: cm per pixel
#         zscale: cm per pixel
#         zstep: number of vertical pixels to move down by per raytrace
#         incidentenergy: x-ray energy (keV) of incoming beam
#         xrfenergy: x-ray energy (keV) of XRF signal
#         sampletheta: angle (degrees) between incident beam and sample normal
#         detectortheta: angle(degrees) between incident beam and XRF detector axis
#     '''
            
#         while in_sample:
#     #         print(f'{k},{i}')

#             if d[i, int(j), int(k)]: #sample exists at coordinate
#                 measured_signal += trace_emission(
#                     i, int(j), int(k), 
#                     mu = mu_emission,
#                     power = incident_power, 
#                     step = step,
#                     theta = detector_theta,
#                     ax = ax
#                 )
#                 incident_power *= step_transmission
#                 if plot:
#                     ax.scatter(int(j), k, c = 'r', s = incident_power*50)
            
#             k -= step_k
#             j += step_j
#             if (k < 0) or (k >= d.shape[2]):
#                 in_sample = False
#             if (i < 0) or (i >= d.shape[0]):
#                 in_sample = False
#             if (int(j) < 0) or (int(j) >= d.shape[1]):
#                 in_sample = False
            

#         return measured_signal
        
    
#     # angles relative to sample plane
#     incident_theta = sampletheta
#     exit_theta = detectortheta - sampletheta

#     d = np.zeros((*thickness.shape, numz)) # d is a 3d mask of sample volume
#     for m,n in np.ndindex(*d.shape[:2]):
#         zidx = int(thickness[m,n]/zscale)
#         d[m,n,:zidx] = 1


#     mu_incident = attenuation_coefficient(elements, numElements, density, incidentenergy)
#     mu_emission = attenuation_coefficient(elements, numElements, density, xrfenergy)
    
#     signal_factor = np.zeros(thickness.shape)
#     for m,n in tqdm(np.ndindex(signal_factor.shape), total = signal_factor.shape[0]*signal_factor.shape[1]):
#         signal_factor[m,n] = trace_incident(m,n, step = zstep , theta = incident_theta)

#     signal_factor /= signal_factor.max()

    
#     return signal_factor

# def trace_emission(i,j,k, mu, power, theta, ax = None):
#     theta = np.deg2rad(theta)
#     step_j = step * np.cos(theta) / zscale_factor
#     step_k = step * np.sin(theta) 
#     step_cm = np.sqrt((step_j*xscale)**2 + (step_k*zscale)**2) * 1e-4
#     step_transmission = np.exp(-mu * step_cm)
    
#     in_sample = True
#     while in_sample:
#         k -= step_k
#         j -= step_j
#         if (k < 0) or (k >= d.shape[2]):
#             in_sample = False
#         if (i < 0) or (i >= d.shape[0]):
#             in_sample = False
#         if (int(j) < 0) or (int(j) >= d.shape[1]):
#             in_sample = False
        
#         if in_sample and d[i, int(j), int(k)]: #sample exists at coordinate
#             power *= step_transmission
#             if ax is not None:
#                 ax.scatter(int(j), k, c = 'b', s = power * 50)
                
#     return power

# def trace_incident(i,j,theta, plot = False):
#     detector_theta = 90 - theta
#     theta = np.deg2rad(theta)
#     step_k = step * np.sin(theta) 
#     step_j = step * np.cos(theta) / zscale_factor
#     step_cm = np.sqrt((step_j*xscale)**2 + (step_k*zscale)**2) * 1e-4
#     step_transmission = np.exp(-mu_incident * step_cm)
    
#     incident_power = 1
#     measured_signal = 0
    
#     k = d.shape[2]-1
#     in_sample = True
    
#     if plot:
#         fig, ax = plt.subplots(figsize = (15,6))
#         ax.imshow(d[i,:,:].T, origin = 'lower')
#         ax.set_aspect(1/zscale_factor)
#         frgplt.scalebar(ax, scale = xscale*1e-6)
#     else:
#         ax = None