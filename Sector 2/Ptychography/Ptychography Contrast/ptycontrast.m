%% ptycontrast.m
%
% Plots the ptychographic phase contrast between materials given their composition, density, thickness, and the energy
% of incident photons.
%
% Input arguments:
%
%   'materials': list of materials in cell format ({'MAPI', 'PbI2', 'Polyimide'})
%   'thicknesses':  list of material thicknesses, in nm
%   'energy':   two element vector with energy range, in eV ([3000, 20000])

function ptycontrast(material_names, thicknesses, energy_range)
    %% user inputs
    % define energy range under consideration
        % energy_range = [3000 20000];                 % energy range to calculate across, in eV. scattering factor data is tabulated from 1000-24900 eV
        num_energies_to_simulate = 5000;             % number of points to divide energy range into for simulation. Tabulated values will be interpolated onto the resulting mesh
        
    % define materials
        % material 1
%         material(1).Name = 'PbF2';               % material name, just used for data presentation
%         material(1).Elements = {'Pb', 'F'};  % list of elements present in material
%         material(1).NumAtoms = [1,2];           % number of each element present in material
%         material(1).Density = 8.44;                 % density in g/cm3
%         material(1).Thickness = 400e-7;             % layer thickness in cm
%         material(1).Type = 2;                       % 1: Target, 2: Background, 3: Substrate
%         
%         %material 2
%         material(2).Name = 'RbPbF3';                 % material name, just used for data presentation
%         material(2).Elements = {'Rb', 'Pb', 'F'};        % list of elements present in material
%         material(2).NumAtoms = [1, 1, 3];              % number of each element present in material
%         material(2).Density = 6.16;                 % density in g/cm3
%         material(2).Thickness = 100e-7;               % layer thickness in cm
%         material(2).Type = 1;                       % 1: Target, 2: Background, 3: Substrate
%         
%         %substrate
%         material(3).Name = 'Polyimide';                 % material name, just used for data presentation
%         material(3).Elements = {'C', 'H', 'N', 'O'};        % list of elements present in material
%         material(3).NumAtoms = [22, 10, 2, 5];              % number of each element present in material
%         material(3).Density = 1.43;                 % density in g/cm3
%         material(3).Thickness = 70e-4;               % layer thickness in cm
%         material(3).Type = 3;                       % 1: Target, 2: Background, 3: Substrate
     
    % find script home directory
        mfilepath = strsplit(mfilename('fullpath'), '\');
        mfilepath = fullfile(mfilepath{1:end-1});
        addpath(genpath(mfilepath));

    % scattering factor database folder
        scatfact_root_folder = fullfile(mfilepath, 'include', 'scatfacts');     % this should point to the folder holding tabulated elemental scattering factor data
       
    % beamline constants
        photon_flux_filepath = fullfile(mfilepath, 'include', 'dummy_beamline_flux.csv');      % this should point to a file with the energy-dependent flux
        sampling_time = 30;     % measurement dwell time, in seconds
   
    % physical constants
        
    r_e = 2.8179403227e-15; %classical radius of electron, m
    r_e = r_e * 1e2;       %convert to cm
    
    h = 4.135667516e-15; %plancks constant in eV/sec 
    
    c = 299792458; %speed of light, m/s
    c = c * 1e2;      %convert to cm
    
    % code start %
    
    %% housekeeping   
    
    material = get_material_data(material_names);       %get material data (density, atomic ratios) from local or Materials Project
    %fill in simulation specific data (thickness)
    for idx = 1:numel(material)
        material(idx).Thickness = thicknesses(idx)*1e-7;     %later math all uses cm, input is in nm, correction factor here
    end
    
    % generate vector of sampling energies/wavelengths
    
    E_photon = linspace(min(energy_range), max(energy_range), num_energies_to_simulate);
    lambda = h*c./E_photon;
    
    
    % check all materials, generate list of all elements in the system and
    % organize by role in the system (target, background, or substrate).
    
    element_list = {};
    
    for mat_idx = 1:numel(material)
        for elem_idx = 1:numel(material(mat_idx).NumAtoms)
            element_list(numel(element_list) + 1) = material(mat_idx).Elements(elem_idx);
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
    
    hfig = figure;
    hold on;

    % Formula 4 %
    phase_shift = zeros(num_energies_to_simulate, numel(material));
    for idx = 1:numel(material)
        phase_shift(:,idx) =  2*pi*material(idx).delta*material(idx).Thickness./lambda;
        plot(E_photon/1000, phase_shift(:,idx));
    end
    
%     total_phase_shift = sum(phase_shift,2);
%     plot(E_photon, total_phase_shift, 'k:');
    title('Ptychography Phase Delay')
    xlabel('Photon Energy (keV)');   
    ylabel('Phase Shift (rad)');    
    xlim([min(E_photon) max(E_photon)]*1e-3);
%     legend(horzcat(material_names, 'Total Shift'));    
    legend(horzcat(material_names));
    
    hcfig = figure;
    hold on;
    him = imagesc(E_photon/1000, 1:size(phase_shift,2), phase_shift');
    xlim([min(E_photon) max(E_photon)]/1000);
    ylim([0.5, size(phase_shift, 2)+0.5]);  %just to align colorbars with plot edges
    hax = him.Parent;
    hax.YTick = 1:size(phase_shift,2);
    hax.YTickLabel = material_names;
    colormap(cbrewer('div', 'Spectral', 256));
    hcb = colorbar;
    title(hcb, 'Phase Shift');
    xlabel('Photon Energy (keV)');
    

    % calculate resolution (this is random junk right now)
    
%     resolution = sqrt((25/8/pi) ./ (photon_flux*sampling_time) .* contrast .* attenuation);    
    
    %% plot results
%     figure, hold on;
    
    % titlestring{1} = 'Total Phase Shift Going Through:';
    % titlestring{2} = sprintf('Target: %.2f nm %s', material(target_idx(1)).Thickness*1e7, material(target_idx(1)).Name);
    % titlestring{3} = sprintf('Background: %.2f nm %s', material(background_idx(1)).Thickness*1e7, material(background_idx(1)).Name);
%     
%     plot(E_photon, total_phase_shift); 
%     title(titlestring);
%     xlabel('Photon Energy (eV)');   
%     ylabel('Phase Shift (rad)');
%        
    
    % figure, hold on;
    
    % titlestring{1} = 'Possible Phase Shifts in Given System:';
    
    % plot(E_photon, phase_shift_target, 'r');
    % plot(E_photon, phase_shift_background, 'b');
    % plot(E_photon, total_phase_shift, 'k');
    
    % legend({material(target_idx(1)).Name, material(background_idx(1)).Name, [material(target_idx(1)).Name ' + ' material(background_idx(1)).Name] , 'Contrast'}, 'location', 'best');
    % title(titlestring);

    
%     prettyplot();
end


function [material_data] = get_material_data(material_names)
    mfilepath = strsplit(mfilename('fullpath'), '\');
    mfilepath = fullfile(mfilepath{1:end-1});
    material_db_path = fullfile(mfilepath, 'include', 'ptychography_material_database.json');       %filepath to our local ptychography material database
%     addpath(fullfile(mfilename('fullpath'), 'include'));
%     disp(material_db_path);
    %open the file, read the single line (json string) and decode to
    %struct, close file
    
    db_fid = fopen(material_db_path, 'r');                      
    db = jsondecode(fgetl(db_fid));
    fclose(db_fid);
    
    found = zeros(size(material_names));    %store whether data for the material has been found yet
    
    fprintf('Looking in local database:\t');
    
    %first look for materials in local database
    for idx = 1:numel(db)
        db_hit = strcmpi(db(idx).Name, material_names);
        if sum(db_hit)
            material_data(db_hit) = db(idx);
            found = found + db_hit;
            fprintf('%s, ', material_names{db_hit});
        end
        if sum(found) == numel(material_names)
            return; %if we found all our data, exit function
        end
        
    end
    %if we are still missing material info, look for it in the materials
    %project database
            
    fprintf('\n\nLooking in the Materials Project database:\t')
            
    for idx = 1:numel(material_names)
        if ~found(idx)
            try
                mp = get_mp('materials', material_names{idx}, 'vasp');  %query the materials project database for a calculated value
            catch
                mp.response = [];
            end
            if ~isempty(mp.response)
                found(idx) = 1;     %mark as found                
                fprintf('%s, ', material_names{idx});   %report finding to console
                
                material_data(idx).Name = material_names{idx};  %populate "Name" field
                
                %populate "Elements" and "NumAtoms" fields
                mp_unitcell = mp.response(1).unit_cell_formula;
                
                material_data(idx).Elements = fieldnames(mp_unitcell)';
                               
                for elem_idx = 1:numel(material_data(idx).Elements)
                    material_data(idx).NumAtoms = mp_unitcell.(material_data(idx).Elements{elem_idx});
                end                
                material_data(idx).Density = mean(vertcat(mp.response.density)); %average all calculated density values
            end
        end
    end
    
    if sum(found) < numel(material_names)
        missing_str = 'Missing material data for some elements:\n ';
        for idx = 1:numel(material_names)
            if ~found(idx)
                addnewquery = input(['\nAdd data for ' material_names{idx} '? (y/n): '], 's');
                if strcmp(addnewquery, 'y')
                    add_to_local_db(material_names{idx});
                else
                    error(['Missing data for ' material_names{idx}]);
                end
            end
        end
    end
    
    fprintf('\n');
end

function add_to_local_db(material_name)
    mfilepath = strsplit(mfilename('fullpath'), '\');
    mfilepath = fullfile(mfilepath{1:end-1});
    material_db_path = fullfile(mfilepath, 'include', 'ptychography_material_database.json');       %filepath to our local ptychography material database
    
    db_fid = fopen(material_db_path, 'r');                      
    db = jsondecode(fgetl(db_fid));
    fclose(db_fid);
    
%     db_newentry = struct(   'Name', {},...
%                             'Elements', {},...
%                             'NumAtoms', [],...
%                             'Density', []...
%                         );
    
    db_newentry.Name = material_name;
    elem_idx = 0;
    add_elems = 1;
    while(add_elems)
        elem_idx = elem_idx + 1;
        data = input(['Element ' num2str(elem_idx) ' (enter blank to stop): '], 's');
        if isempty(data)
            add_elems = 0;
            break;
        else
            db_newentry.Elements{elem_idx} = data;
            db_newentry.NumAtoms(elem_idx) = input(['# of ' data ' atoms per unit: ']);
        end
    end
    db_newentry.Density = input('Material Density (g/cm^3): ');
    
    db = [db; db_newentry];    
    db_writestring = jsonencode(db);
    
    db_fid = fopen(material_db_path, 'w');                      
    fprintf(db_fid, db_writestring);
    fclose(db_fid);
end