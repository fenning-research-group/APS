import os
import matplotlib.pyplot as plt
import numpy as np
import csv


packageDir = os.path.dirname(os.path.abspath(__file__))

def ScatteringFactor(element, energy):
	dataDir = os.path.join(packageDir, 'include', 'scatfacts')
#     dataElements = [x[:-4] for x in os.listdir(dataDir)]

	fid = os.path.join(dataDir, '{0}.nff'.format(str.lower(element)))
	with open(fid, 'r') as f:
		data = np.genfromtxt(f)[1:, :]
	e_raw = data[:, 0]
	f1_raw = data[:, 1]
	f2_raw = data[:, 2]

	f1 = np.interp(energy, e_raw, f1_raw)
	f2 = np.interp(energy, e_raw, f2_raw)
	return f1, f2

def MolarMass(element):
	dataPath = os.path.join(packageDir, 'include', 'Molar Masses.txt')
	molarMass = None
	with open(dataPath, 'r') as f:
		reader = csv.reader(f, delimiter=',', quotechar='|')
		for row in reader:
			if row[0] == element:
				molarMass = float(row[1])
				break
	return molarMass


def AttenuationCoefficient(elements, numElements, density, energy):
	N_a = 6.022e23
	c = (1e-16)/(np.pi*0.9111)  # eV*cm^2
	energy = np.array(energy)
	f2 = 0
	mass = 0
	for i, el, num in zip(range(len(elements)), elements, numElements):
		_, f2temp = ScatteringFactor(el, energy)
		f2 = f2 + num*f2temp
		mass = mass + num*MolarMass(el)

	mu = (density*N_a/mass) * (2*c/energy) * f2
	return mu

def AttenuationLength(elements, numElements, density, energy):
	mu = AttenuationCoefficient(elements, numElements, density, energy)
	return 1/mu

def Transmission(elements, numElements, density, thickness, energy):
	mu = AttenuationCoefficient(
		elements, numElements, density, energy)

	t = np.exp(-mu*thickness)

	return t

def XRFSelfAbsorption(elements, numElements, density, thickness, incidentenergy, xrfenergy, sampletheta, detectortheta):
	incidentAttCoeff = AttenuationCoefficient(elements, numElements, density, incidentenergy)
	exitAttCoeff = AttenuationCoefficient(elements, numElements, density, xrfenergy)
	
	incidentTheta = sampletheta
	exitTheta = detectortheta - sampletheta

	c = (incidentAttCoeff/np.cos(np.deg2rad(incidentTheta))) + (exitAttCoeff/np.cos(np.deg2rad(exitTheta)))

	xrfFraction = (1/thickness) * (1/c) * (1 - np.exp(-c * thickness))

	return xrfFraction

