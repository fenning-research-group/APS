function ptycontrast()
    %% user inputs
    
    % define energy range under consideration
        energy_range = [3000 20000];                 % energy range to calculate across, in eV. scattering factor data is tabulated from 1000-24900 eV
        num_energies_to_simulate = 1000;             % number of points to divide energy range into for simulation. Tabulated values will be interpolated onto the resulting mesh
        
    % define materials
        % material 1
        material(1).Name = 'PbF2';               % material name, just used for data presentation
        material(1).Elements = {'Pb', 'F'};  % list of elements present in material
        material(1).NumAtoms = [1,2];           % number of each element present in material
        material(1).Density = 8.44;                 % density in g/cm3
        material(1).Thickness = 400e-7;             % layer thickness in cm
        material(1).Type = 2;                       % 1: Target, 2: Background, 3: Substrate
        
        %material 2
        material(2).Name = 'RbPbF3';                 % material name, just used for data presentation
        material(2).Elements = {'Rb', 'Pb', 'F'};        % list of elements present in material
        material(2).NumAtoms = [1, 1, 3];              % number of each element present in material
        material(2).Density = 6.16;                 % density in g/cm3
        material(2).Thickness = 100e-7;               % layer thickness in cm
        material(2).Type = 1;                       % 1: Target, 2: Background, 3: Substrate
        
        %substrate
        material(3).Name = 'Polyimide';                 % material name, just used for data presentation
        material(3).Elements = {'C', 'H', 'N', 'O'};        % list of elements present in material
        material(3).NumAtoms = [22, 10, 2, 5];              % number of each element present in material
        material(3).Density = 1.43;                 % density in g/cm3
        material(3).Thickness = 70e-4;               % layer thickness in cm
        material(3).Type = 3;                       % 1: Target, 2: Background, 3: Substrate
        
    % scattering factor database folder
        scatfact_root_folder = 'G:\My Drive\FRG\Projects\APS\Scripts\Ptychography Contrast\scatfacts';     % this should point to the folder holding tabulated elemental scattering factor data
       
    % beamline constants
        photon_flux_filepath = 'G:\My Drive\FRG\Projects\APS\Scripts\Ptychography Contrast\dummy_beamline_flux.csv';      % this should point to a file with the energy-dependent flux
        sampling_time = 30;     % measurement dwell time, in seconds
   
    % physical constants
        
    r_e = 2.8179403227e-15; %classical radius of electron, m
    r_e = r_e * 1e2;       %convert to cm
    
    h = 4.135667516e-15; %plancks constant in eV/sec 
    
    c = 299792458; %speed of light, m/s
    c = c * 1e2;      %convert to cm
    
    % code start %
    
    %% housekeeping
    
    % generate vector of sampling energies/wavelengths
    
    E_photon = linspace(min(energy_range), max(energy_range), num_energies_to_simulate);
    lambda = h*c./E_photon;
    
    
    % check all materials, generate list of all elements in the system and
    % organize by role in the system (target, background, or substrate).
    
    target_idx = [];
    background_idx = [];
    substrate_idx = [];
    element_list = {};
    
    for mat_idx = 1:numel(material)
        for elem_idx = 1:numel(material(mat_idx).NumAtoms)
            element_list(numel(element_list) + 1) = material(mat_idx).Elements(elem_idx);
        end
        
        switch material(mat_idx).Type
            case 1
                target_idx = [target_idx, mat_idx];
            case 2
                background_idx = [background_idx, mat_idx];
            case 3
                substrate_idx = [substrate_idx, mat_idx];
        end
    end 
    
    element_list = unique(element_list);        % remove duplicates
    
    % read beamline photon flux file
    photon_flux = csvread(photon_flux_filepath);
    photon_flux = interp1(photon_flux(:,1), photon_flux(:,2), E_photon);        % generates vector of photon flux, interpolated to the simulation energy values   %FIX: catch/some solution for simulation outside energy range covered by photon flux file%
    
    %% pull scattering factors (f, f') from tables. load elemental densities and atomic masses as well
       
    for element_name = element_list
        element_name = element_name{1};
        fid = fopen(fullfile(scatfact_root_folder, [element_name '.nff']), 'r');
        fgetl(fid);
        
        raw_scatfact_data = textscan(fid, '\t%f\t%f\t%f'); %each cell holds {E_photon eV, Real Scattering Component f', Imaginary Scattering Component f")
        element.(element_name).f_real = interp1(raw_scatfact_data{1}, raw_scatfact_data{2}, E_photon);      %interpolate real scattering factor to simulation energy values
        element.(element_name).f_imaginary = interp1(raw_scatfact_data{1}, raw_scatfact_data{3}, E_photon); %interpolate imaginary scattering factor to simulation energy values
        
        for energy_idx = 1:numel(E_photon)
            [~, mucal_output] = mucal(element_name, 0, (E_photon(energy_idx)/1000), 0, 0);  %mucal takes energy input in keV, rest of program is using eV
            element.(element_name).total_cross_section(energy_idx) = mucal_output(4);     %total cross section, in cm^2/g
            element.(element_name).mu(energy_idx) = mucal_output(6);                      %attenuation coefficient, in cm^-1
        end
        
        element.(element_name).Density = mucal_output(8);
        element.(element_name).AtomicMass = mucal_output(7);
        fclose(fid);
    end
    
    %% calculate delta (real index of refraction decrement) and absorption coefficient mu (imaginary index of refraction) for each material

    for mat_idx = 1:numel(material)
        e_scatter_summation = zeros(1, num_energies_to_simulate);
        material_component_total_xsec = zeros(1, num_energies_to_simulate);  %used as intermediate step in attenuation coefficient calculation, not a physical value!
        material_molar_mass = 0;      %total molar mass of material
        
        
        % Formula 1 %
        for elem_idx = 1:numel(material(mat_idx).NumAtoms)
            material_molar_mass = material_molar_mass + element.(material(mat_idx).Elements{elem_idx}).AtomicMass * material(mat_idx).NumAtoms(elem_idx);   
        end
        
        for elem_idx = 1:numel(material(mat_idx).NumAtoms)
            % Formula 2.1 %
            e_scatter_summation = e_scatter_summation + element.(material(mat_idx).Elements{elem_idx}).f_real * 6.022e23 * material(mat_idx).NumAtoms(elem_idx) * material(mat_idx).Density / material_molar_mass;    %calcualting summation of N_atoms (atoms/cm3) * forward scatter real refractive index for each atomic species in material              
            % Formula 3.1 %
            material_component_total_xsec = material_component_total_xsec + element.(material(mat_idx).Elements{elem_idx}).mu * material(mat_idx).NumAtoms(elem_idx);   %calcualting summation of N_atoms * forward scatter imaginary refractive index for each atomic species in material
        end
        
        % Formula 2.2 %
        material(mat_idx).delta = (r_e/2/pi) * e_scatter_summation .* (lambda.^2);
        
        % Formula 3.2 %
        material(mat_idx).mu = (6.022e23 * 1e-24) * material(mat_idx).Density * material_component_total_xsec ./ material_molar_mass;      %divide sum of absorption coefficients by total number of atoms, making this a weighted average absorption coefficient
    end
    
    
    %% calculate phase contrast
    
    % calculate attenuation from absorption within materials
%     attenuation = 0;
%     for i = 1:numel(material)
%         attenuation = attenuation + material(i).mu * material(i).Thickness;
%     end
% 
%     attenuation = exp(-attenuation);
%     
    % calculate contrast term  
    
    % Formula 4 %

    total_phase_shift = 2*pi*((material(target_idx(1)).delta*material(target_idx(1)).Thickness + material(background_idx(1)).delta*material(background_idx(1)).Thickness))./lambda;
    
    
    % Formula 5 %
    phase_shift_target = 2*pi*material(target_idx(1)).delta*material(target_idx(1)).Thickness./lambda;
    phase_shift_background = 2*pi*material(background_idx(1)).delta*material(background_idx(1)).Thickness./lambda;
    
    % calculate resolution (this is random junk right now)
    
%     resolution = sqrt((25/8/pi) ./ (photon_flux*sampling_time) .* contrast .* attenuation);    
    
    %% plot results
%     figure, hold on;
    
    titlestring{1} = 'Total Phase Shift Going Through:';
    titlestring{2} = sprintf('Target: %.2f nm %s', material(target_idx(1)).Thickness*1e7, material(target_idx(1)).Name);
    titlestring{3} = sprintf('Background: %.2f nm %s', material(background_idx(1)).Thickness*1e7, material(background_idx(1)).Name);
%     
%     plot(E_photon, total_phase_shift); 
%     title(titlestring);
%     xlabel('Photon Energy (eV)');   
%     ylabel('Phase Shift (rad)');
%        
    
    figure, hold on;
    
    titlestring{1} = 'Possible Phase Shifts in Given System:';
    
    plot(E_photon, phase_shift_target, 'r');
    plot(E_photon, phase_shift_background, 'b');
    plot(E_photon, total_phase_shift, 'k');
    
    legend({material(target_idx(1)).Name, material(background_idx(1)).Name, [material(target_idx(1)).Name ' + ' material(background_idx(1)).Name] , 'Contrast'}, 'location', 'best');
    title(titlestring);
    xlabel('Photon Energy (eV)');   
    ylabel('Phase Shift (rad)');
    
%     prettyplot();
end