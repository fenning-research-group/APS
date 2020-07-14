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

def add_XRF_lines(elements, ax = None, maxlinesperelement = np.inf, tickloc = 'bottom', tickstagger = 0, ticklength = 0.05):     
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
    dataPath = os.path.join(packageDir, 'include', 'Molar Masses.txt')
    molar_mass = None
    with open(dataPath, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            if row[0] == element:
                molar_mass = float(row[1])
                break
    return molar_mass

### Attenuation, transmission, self-absorption, etc.

def attenuation_coefficient(elements, numElements, density, energy):
    '''
    returns x-ray attenuation coefficient, in cm-1, given:
        elements: list of elements ['Fe', 'Cu']
        numElements: list of numbers corresponding to elemental composition. [1,2] for FeCu2
        density: overall density of material (g/cm3)
        energy: x-ray energy(ies) (keV)
    '''
    N_a = 6.022e23
    c = (1e-19)/(np.pi*0.9111)  # keV*cm^2
    energy = np.array(energy)
    f2 = 0
    mass = 0
    for i, el, num in zip(range(len(elements)), elements, numElements):
        _, f2temp = scattering_factor(el, energy)
        f2 = f2 + num*f2temp
        mass = mass + num*molar_mass(el)

    mu = (density*N_a/mass) * (2*c/energy) * f2
    return mu

def attenuation_length(elements, numElements, density, energy):
    '''
    returns x-ray attenuation length (distance for transmitted intensity to
    decay to 1/e), in cm
    '''
    mu = attenuation_coefficient(elements, numElements, density, energy)
    return 1/mu

def transmission(elements, numElements, density, thickness, energy):
    '''
    returns fraction of x-ray intensity transmitted through a sample, defined by

        elements: list of elements ['Fe', 'Cu']
        numElements: list of numbers corresponding to elemental composition. [1,2] for FeCu2
        density: overall density of material (g/cm3)
        thickness: path length of x rays (cm)
        energy: x-ray energy (keV)
    '''

    mu = attenuation_coefficient(
        elements, numElements, density, energy)

    t = np.exp(-mu*thickness)

    return t

def self_absorption(elements, numElements, density, thickness, incidentenergy, xrfenergy, sampletheta, detectortheta):
    '''
    returns fraction of x-ray fluorescence excited, transmitted through a sample, and reaching to an XRF detector. This
    calculation assumes no secondary fluorescence/photon recycling. The returned fraction is the apparent signal after 
    incident beam attenuation and exit fluorescence attenuation - dividing the measured XRF value by this fraction should
    approximately correct for self-absorption losses and allow better comparison of fluorescence signals in different energy
    ranges.

    Calculations are defined by:

        elements: list of elements ['Fe', 'Cu']
        numElements: list of numbers corresponding to elemental composition. [1,2] for FeCu2
        density: overall density of material (g/cm3)
        thickness: Sample thickness - NOT PATH LENGTH (cm)
        incidentenergy: x-ray energy (keV) of incoming beam
        xrfenergy: x-ray energy (keV) of XRF signal
        sampletheta: angle (degrees) between incident beam and sample normal
        detectortheta: angle(degrees) between incident beam and XRF detector axis
    '''

    incidentAttCoeff = attenuation_coefficient(elements, numElements, density, incidentenergy)
    exitAttCoeff = attenuation_coefficient(elements, numElements, density, xrfenergy)
    
    incidentTheta = sampletheta
    exitTheta = detectortheta - sampletheta

    c = (incidentAttCoeff/np.cos(np.deg2rad(incidentTheta))) + (exitAttCoeff/np.cos(np.deg2rad(exitTheta)))

    xrfFraction = (1/thickness) * (1/c) * (1 - np.exp(-c * thickness))

    return xrfFraction

