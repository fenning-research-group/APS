%% densityfromlattice.m
%


function density = densityfromlattice(element_list, element_counts, lattice_parameters)
    %% user inputs
       
    % find script home directory
        mfilepath = strsplit(mfilename('fullpath'), '\');
        mfilepath = fullfile(mfilepath{1:end-1});
        addpath(genpath(mfilepath));

    % scattering factor database folder
        scatfact_root_folder = fullfile(mfilepath, 'include', 'scatfacts');     % this should point to the folder holding tabulated elemental scattering factor data
 
    % physical constants
        
        r_e = 2.8179403227e-15; %classical radius of electron, m
        r_e = r_e * 1e2;       %convert to cm

        h = 4.135667516e-15; %plancks constant in eV/sec 

        c = 299792458; %speed of light, m/s
        c = c * 1e2;      %convert to cm
        
        Na = 6.0221409e23;      %Avogadro's number, atoms/mol
        
    % code start %
    

    %% Determine molar mass of one unit of material
    
    molarmass = 0;
    
    for idx = 1:length(element_counts)
        element_name = element_list{idx};
        numatoms = element_counts(idx);
                
        [~, mucal_output] = mucal(element_name, 0, 1, 0, 0);  %mucal takes energy input in keV, rest of program is using eV
        
        atomicmass = mucal_output(7);
        
        molarmass = molarmass +  atomicmass*numatoms;
    end
    
    %% Determine volume of unit cell (note: assumes all 90 degree angles!)
        
    v_unitcell = lattice_parameters(1)*lattice_parameters(2)*lattice_parameters(3) * 1e-24;     %convert volume from cubic angstoms to cubic cm
    
    density = molarmass / Na / v_unitcell;
    
end
    