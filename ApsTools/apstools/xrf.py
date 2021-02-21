import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
import matplotlib.transforms as transforms
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.pool as mpp
from scipy.ndimage import center_of_mass

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
        Na = 6.022e23
        c = (1e-19)/(np.pi*0.9111)  # keV*cm^2
        energy = np.array(energy)
        f2 = 0
        mass = 0
        for i, (el, num) in enumerate(self.elements.items()):
            _, f2_ = scattering_factor(el, energy)
            f2 += num*f2_
            mass += num*molar_mass(el)

        mu = (self.density*Na/mass) * (2*c/energy) * f2
        return mu
    
    def attenuation_length(self, energy):
        '''
        returns x-ray attenuation length (distance for transmitted intensity to
        decay to 1/e), in cm
        '''
        mu = self.attenuation_coefficient(energy)
        return 1/mu

    def transmission(self, thickness, energy):
        '''
        returns fraction of x-ray intensity transmitted through a sample, defined by
            thickness: path length of x rays (cm)
            energy: x-ray energy (keV)
        '''

        mu = self.attenuation_coefficient(energy)
        t = np.exp(-mu*thickness)

        return t

    def phase_delay(self, thickness, energy):
        '''
        calculates phase delay (radians) of photons passing through material slab.
        useful for predicting phase contrast in ptychography measurements.
        '''
        r_e = 2.8179403227e-13; #classical radius of electron, cm            
        h = 4.135667516e-18; #plancks constant, keV/sec 
        c = 299792458e2; #speed of light, cm/s
        Na = 6.022e23 #avogadros number, atoms/mol

        wl = h*c/np.asarray(energy) #photon wavelengths, cm

        f1 = 0
        mass = 0
        for el, num in self.elements.items():
            f1_, _ = scattering_factor(el, energy)
            f1 += num*f1_
            mass += num*molar_mass(el)

        delta = f1*(self.density*Na/mass) * (r_e/2/np.pi) * (wl**2)
        phase_delay = 2*np.pi*delta*thickness/wl

        return phase_delay

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

class ParticleXRF:
    def __init__(self, z, scale, sample_theta, detector_theta):
        # self.material = material
        self.z = z
        self.scale = scale
        self.sample_theta = sample_theta
        self.detector_theta = detector_theta
        self.align_method = 'com'
        self._interaction_weight = None

    @property
    def scale(self):
        return self.__scale
    @scale.setter
    def scale(self, scale):
        scale = np.array(scale)
        if len(scale) == 1:
            scale = np.tile(scale, 3) #single scale value given, assume same for x,y,z
        self.__scale = scale

    @property
    def sample_theta(self):
        return self.__sample_theta
    @sample_theta.setter
    def sample_theta(self, sample_theta):
        self.__sample_theta = sample_theta
        self.__sample_theta_rad = np.deg2rad(sample_theta)

    @property
    def detector_theta(self):
        return self.__detector_theta
    @detector_theta.setter
    def detector_theta(self, detector_theta):
        self.__detector_theta = detector_theta
        self.__detector_theta_rad = np.deg2rad(detector_theta)

    @property
    def align_method(self):
        return self.__align_method
    @align_method.setter
    def align_method(self, m):
        if m in [0,'leading']:
            self.__align_method = 'leading'
        elif m in [1, 'com']:
            self.__align_method = 'com'
        else:
            raise ValueError('.align_method must be \'leading\' or \'com\'')

    def __check_if_in_simulation_bounds(self, i, j, k):
        for current_position, simulation_bound in zip([i,j,k], self._d.shape):
            if (current_position < 0) or (int(round(current_position)) >= simulation_bound):
                return False
        return True

    def raytrace(self, step = 5, pad_left = None, pad_right = None, chunksize = 200, n_workers = 4):
        self.step = step
        self.__step_cm = step * self.scale[2] * 1e-4 # step in units of k (z axis)
        
        pad_default = np.ceil(
            np.abs((self.z.max() / np.tan(self.__sample_theta_rad))) #max x displacement based off max z 
            /self.scale[1]                   #scaled in case x and z scales differ
            ).astype(int) + 1 #add 1 to buffer and avoid clipping

        if pad_left is None:
            pad_left = pad_default
        if pad_right is None:
            pad_right = pad_default

        self.__pad_left = pad_left
        self.__pad_right = pad_right
        x_pad = pad_right + pad_left
        numz = int(self.z.max()/ self.scale[2])
        
        self._d = np.full((self.z.shape[0], self.z.shape[1]+x_pad, numz), False) # 3d boolean mask of sample volume - assumes no embedded holes in sample
        for i,j in np.ndindex(*self.z.shape):
            zidx = int(self.z[i,j] / self.scale[2])
            self._d[i,int(j+pad_left),:zidx] = True

        self._incident_steps = [[None for j in range(self._d.shape[1])]
                                      for i in range(self._d.shape[0])]
        self._emission_steps = [[None for j in range(self._d.shape[1])]
                                      for i in range(self._d.shape[0])]

        with mp.Pool(n_workers) as pool:
            pts = list(np.ndindex(self._d.shape[:2]))
            raytrace_output = []
            for output in tqdm(pool.istarmap(self._trace_incident, pts, chunksize = chunksize), total = len(pts)):
                raytrace_output.append(output)

            # raytrace_output = pool.starmap(self._trace_incident, pts, chunksize = chunksize)
        for (i,j), (in_pts, em_steps) in zip(pts, raytrace_output):
            self._incident_steps[i][j] = in_pts
            self._emission_steps[i][j] = em_steps

    def calc_ssf(self, material, incident_energy):
        self.ssf = []
        self._interaction_weight = np.zeros(self._d.shape[:2])
        step_transmission = material.transmission(self.__step_cm, incident_energy)

        for i, row in enumerate(self._incident_steps):
            row_ssf = []
            for j, ray in enumerate(row):
                this_point = {'coordinate':[], 'weight':[]}
                incident_power = 1
                for intersection_pt in ray:
                    this_point['coordinate'].append(intersection_pt)
                    this_point['weight'].append(incident_power)
                    incident_power *= step_transmission
                # this_point['weight'] = np.array(this_point['weight'])
                this_point['coordinate'] = np.array(this_point['coordinate'])
                total_weight = np.sum(this_point['weight'])
                this_point['weight'] = np.array([w/total_weight for w in this_point['weight']]) #normalize interaction weight so all points sum to 1
                row_ssf.append(this_point)
                self._interaction_weight[i,j] = total_weight
            self.ssf.append(row_ssf)
        self.ssf = self._clip_to_original(np.array(self.ssf), ssf = True)

        return self.ssf

    def calc_self_absorption_factor(self, material, incident_energy, emission_energy):
        if self._interaction_weight is None:
            self.calc_ssf(material, incident_energy)

        self.saf = np.zeros(self._d.shape[:2])

        step_transmission_incident = material.transmission(self.__step_cm, incident_energy)
        step_transmission_emission = material.transmission(self.__step_cm, emission_energy)

        for i, (incident_row, emission_row) in enumerate(zip(self._incident_steps, self._emission_steps)):
            for j, (incident_ray, exit_ray) in enumerate(zip(incident_row, emission_row)):
                incident_power = 1
                emission_power = 0
                for intersection_pt, exit_pts in zip(incident_ray, exit_ray):
                    abs_power = incident_power * (1-step_transmission_incident) #assume all power attenuated at this step is absorbed + fluoresced
                    emission_power += abs_power * (step_transmission_emission ** exit_pts) #attenuate fluoresced signal for all intersection steps on the exit path
                    incident_power *= step_transmission_incident #attenuate incident beam before moving on to next intersection point
                self.saf[i,j] = emission_power
        self.saf = self._clip_to_original(np.array(self.saf))

        return self.saf

    def _clip_to_original(self, x, ssf = False):
        offset_method_lookup = {
            'leading': self._find_offset_leading_edge,
            'com':     self._find_offset_com
        }
        offset_method = offset_method_lookup[self.__align_method]
        alignment_offset = offset_method()

        j_start = self.__pad_left - alignment_offset
        slice_j = slice(
            j_start,
            j_start + self.z.shape[1],
            1
        )

        if ssf:
            for m,n in np.ndindex(x.shape):
                if len(x[m,n]['coordinate']):
                    x[m,n]['coordinate'][:,1] -= (j_start+1) #decrement by one more to account for zero indexing

        return x[:, slice_j]


    def _find_offset_leading_edge(self, threshold = 0.1):
        def find_offset_line(i):
            z_line = np.argmin(self._d[i], axis = 1)
            z_above_threshold = np.where(z_line > d_thresh)[0]
            if len(z_above_threshold) == 0:
                return np.nan, np.nan
            x_z_min = z_above_threshold.min()
            x_z_max = z_above_threshold.max()
            
            signal_factor = self._interaction_weight[i]
            signal_above_threshold = np.where(signal_factor > signal_thresh)[0]
            if len(signal_above_threshold) == 0:
                return np.nan, np.nan
            x_signal_min = signal_above_threshold.min()
            x_signal_max = signal_above_threshold.max()

            left_edge_offset = x_z_min - x_signal_min
            right_edge_offset =  x_z_max - x_signal_max

            return left_edge_offset, right_edge_offset

        d_thresh = self._d.max() * threshold
        signal_thresh = self._interaction_weight.max() * threshold

        if np.abs(self.sample_theta) <= 90: #beam enters from right side of sample
            offset = np.nanmean([find_offset_line(i)[1] for i in range(self._d.shape[0])])
        else:
            offset = np.nanmean([find_offset_line(i)[0] for i in range(self._d.shape[0])])

        return int(round(offset)) #need to clip the output arrays at integer index values
    
    def _find_offset_com(self):
        z_com = center_of_mass(self._d.argmin(axis = 2))
        signal_com = center_of_mass(self._interaction_weight)
        offset = z_com[1] - signal_com[1] #only offset in x from beam projection - pencil beam contained in xz plane

        return int(round(offset))


        return offset
    def _trace_emission(self, i,j,k):
        exit_theta = self.__sample_theta_rad - self.__detector_theta_rad
        step_i = 0 * (self.scale[2]/self.scale[1]) #step is in units of k (z axis)
        step_j = self.step * np.cos(exit_theta) * (self.scale[2]/self.scale[0]) #step is in units of k (z axis)
        step_k = self.step * np.sin(exit_theta) 
        n_attenuation_steps = 0

        in_bounds = True
        while in_bounds:
            i += step_i
            j += step_j
            k -= step_k

            i_ = int(round(i))
            j_ = int(round(j))
            k_ = int(round(k))
            in_bounds = self.__check_if_in_simulation_bounds(i,j,k)                        

            if in_bounds and self._d[i_, j_, k_]: #sample exists at coordinate
                n_attenuation_steps += 1

        return n_attenuation_steps

    def _trace_incident(self, i,j):
        step_i = 0 * (self.scale[2]/self.scale[1]) #step is in units of k (z axis)
        step_j = self.step * np.cos(self.__sample_theta_rad) * (self.scale[2]/self.scale[0]) #step is in units of k (z axis)
        step_k = self.step * np.sin(self.__sample_theta_rad) 
        
        attenuation_points = []
        n_emission_steps = []
        
        k = self._d.shape[2]-1

        in_bounds = True
        while in_bounds:
            i_ = int(round(i))
            j_ = int(round(j))
            k_ = int(round(k))

            if self._d[i_, j_, k_]: #sample exists at coordinate
                attenuation_points.append((i_, j_, k_))
                n_emission_steps.append(self._trace_emission(i, j, k))
            
            i -= step_i
            j -= step_j
            k -= step_k
            in_bounds = self.__check_if_in_simulation_bounds(i,j,k)                        

        return attenuation_points, n_emission_steps

# step_cm = np.sqrt((step_j*xscale)**2 + (step_k*zscale)**2) * 1e-4
#         step_transmission = np.exp(-mu_incident * step_cm)



# def self_absorption_particle(z, scale, material, incident_energy, emission_energy, sample_theta, detector_theta = 90, step = 20, zscale_factor = 40, z_background_threshold = 0.2):
#     def trace_emission(i,j,k, mu, power, step = step, theta = 90, ax = None):
#         theta = np.deg2rad(theta)
#         step_j = step / zscale_factor * np.cos(theta)
#         step_k = step * np.sin(theta) 
#         step_cm = np.sqrt((step_j*xscale)**2 + (step_k*zscale)**2) * 1e-4
#         step_transmission = np.exp(-mu * step_cm)
#         in_sample = True
#         while in_sample:
#             k -= step_k
#             j += step_j
#             if (k < 0) or (int(round(k)) >= d.shape[2]):
#                 in_sample = False
#             if (i < 0) or (i >= d.shape[0]):
#                 in_sample = False
#             if (int(round(j)) < 0) or (int(round(j)) >= d.shape[1]):
#                 in_sample = False

#             if in_sample and d[i, int(round(j)), int(round(k))]: #sample exists at coordinate
#                 power *= step_transmission
#                 if ax is not None:
#                     ax.scatter(j, k, c = 'b', s = power * 50)

#         return power
    
#     def trace_incident(i,j, mu_incident, mu_emission, step = step, theta = 90, detector_theta = 90, plot = False):
#         exit_theta = detector_theta - theta
#         theta = np.deg2rad(theta)
#         step_k = step * np.sin(theta) 
#         step_j = step / zscale_factor * np.cos(theta)
#         step_cm = np.sqrt((step_j*xscale)**2 + (step_k*zscale)**2) * 1e-4
#         step_transmission = np.exp(-mu_incident * step_cm)

#         incident_power = 1
#         measured_signal = 0
#         kernel = {key:[] for key in ['coordinate', 'weight']}
        
#         k = d.shape[2]-1
#         in_sample = True

#         if plot:
#             fig, ax = plt.subplots(figsize = (15,6))
#             ax.imshow(d[i,:,:].T, origin = 'lower')
#             ax.set_aspect(1/zscale_factor)
#             frgplt.scalebar(scale = xscale*1e-6, ax = ax)
#         else:
#             ax = None

#         while in_sample:
#     #         print(f'{k},{i}')
#             if d[i, int(round(j)), int(round(k))]: #sample exists at coordinate
#                 measured_signal += trace_emission(
#                     i, int(j), int(k), 
#                     mu = mu_emission,
#                     power = incident_power*(1-step_transmission), 
#                     step = step,
#                     theta = exit_theta,
#                     ax = ax
#                 )
#                 kernel['coordinate'].append([i, int(round(j))])
#                 kernel['weight'].append(incident_power)
#                 if plot:
#                     ax.scatter(j, k, c = 'r', s = incident_power*50)
                
#                 incident_power *= step_transmission


#             k -= step_k
#             j -= step_j
#             if (k < 0) or (int(round(k)) >= d.shape[2]):
#                 in_sample = False
#             if (i < 0) or (i >= d.shape[0]):
#                 in_sample = False
#             if (int(round(j)) < 0) or (int(round(j)) >= d.shape[1]):
#                 in_sample = False

#         kernel = {key: np.array(val) for key, val in kernel.items()}
#         kernel['weight'] = kernel['weight'] / kernel['weight'].sum()

#         return measured_signal, kernel
        
#     ## Mesh generation
#     xscale = yscale = zscale = scale
#     zscale /= zscale_factor
#     x_pad_left = 50
#     x_pad_right = 250
#     x_pad = x_pad_right + x_pad_left
    
#     z_background = z[z < z_background_threshold] #consider anything under 200 nm to be background/not crystal
#     z -= z_background.mean()
#     z[z < 0] = 0
#     numz = int(z.max()/ zscale)
    
#     # 3d boolean mask of sample volume - assumes no embedded holes in sample
#     d = np.full((z.shape[0], z.shape[1]+x_pad, numz), False)
#     for m,n in np.ndindex(*z.shape):
#         zidx = int(z[m,n]/zscale)
#         d[m,int(n+x_pad_left),:zidx] = True
    
#     ## simulation setup
#     mu_incident = xrft.attenuation_coefficient(
#         energy = incident_energy,
#         **material
#     )
    
#     if type(emission_energy) is not list:
#         emission_energy = [emission_energy]
    
    
#     output = {
#         'signal_factor': {},
#         'incident_energy': incident_energy,
#     }
#     kernels = [[[] for n in range(d.shape[1])]
#                    for m in range(d.shape[0])]
    
#     generate_kernels = True
#     for emission_energy_ in emission_energy:
#         mu_emission = xrft.attenuation_coefficient(
#             energy = emission_energy_,
#             **material
#         )
        
#         output['signal_factor'][emission_energy_] = np.zeros(d.shape[:2])

#         for m,n in tqdm(np.ndindex(d.shape[:2]), total = d.shape[0]*d.shape[1]):
#             output['signal_factor'][emission_energy_][m,n], kernels[m][n] = trace_incident(
#                 m,
#                 n,
#                 mu_incident = mu_incident,
#                 mu_emission = mu_emission,
#                 step = step,
#                 theta = sample_theta,
#                 detector_theta = detector_theta,
#             )
    
#     x_lineup_offset_com = -int(round(center_of_mass(d.argmin(axis = 2))[1] - center_of_mass(output['signal_factor'][emission_energy_])[1]))
#     x_start = x_pad_left + x_lineup_offset_com
#     x_end = x_start + z.shape[1]
#     x_offset_array = np.array([0, -x_start])
    
#     for emission_energy_ in emission_energy:
#         output['signal_factor'][emission_energy_] = output['signal_factor'][emission_energy_][:, x_start:x_end]
    
#     output['extent'] = [0, output['signal_factor'][emission_energy_].shape[1]*xscale, 0, output['signal_factor'][emission_energy_].shape[0]*yscale]
#     output['z'] = z
#     output['kernels'] = [k[x_start:x_end] for k in kernels]
#     for m, k0 in enumerate(output['kernels']):
#         for n, k1 in enumerate(k0):
#             if len(output['kernels'][m][n]['coordinate']) > 0:
#                 output['kernels'][m][n]['coordinate'] += x_offset_array
    
#     return output




###### https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
### Python 3.8+
# def __istarmap(self, func, iterable, chunksize=1):
#     """starmap-version of imap
#     """
#     self._check_running()
#     if chunksize < 1:
#         raise ValueError(
#             "Chunksize must be 1+, not {0:n}".format(
#                 chunksize))

#     task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
#     result = mpp.IMapIterator(self)
#     self._taskqueue.put(
#         (
#             self._guarded_task_generation(result._job,
#                                           mpp.starmapstar,
#                                           task_batches),
#             result._set_length
#         ))
#     return (item for chunk in result for item in chunk)

def __istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = __istarmap