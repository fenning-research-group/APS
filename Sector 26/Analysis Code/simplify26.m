function out = simplify26(scannum, qbin)
    %% Inputs
    
    % Single Element ROI Designations
        % which element is each ROI set to? 
    
    roi_numbers = [6, 9, 18, 24, 26, 17];
    roi_labels = {'Cs', 'Pb', 'Br', 'Cr', 'Ca', 'I/Eu'};
    
    
    %% detector identification 
    pilatus_name = 'dp_pilatusASD:cam1:FileNumber_RBV';
    single_elem_detector_name = 'mca8.R';

    %% generate two theta map for pilatus pixels
    twothlim = [min(qbin.twotheta(:)), max(qbin.twotheta(:))];
    twoth_tolerance = ( max(qbin.twotheta(1,:)) - min(qbin.twotheta(1,:)) ) / size(qbin.twotheta, 2) / 2;    %what value to round the 2theta values to - set to pixel width in 2theta
    num_bins = round((twothlim(2) - twothlim(1))/(2*twoth_tolerance));
    twoth = linspace(twothlim(1), twothlim(2), num_bins);       %two theta values onto which the pilatus images will be collapsed
    
    %% load and evaluate mda file
    dat = mdaload(['mda/26idbSOFT_' num2str(scannum,'%4.4d') '.mda']);
    pt = dat.scan(1).sub_scans(1);  %single point to get detector channels
    
    % size of scan
    numpts_m = dat.dimensions(1);
    numpts_n = dat.dimensions(2);
    
    % identify detector channels
    
    single_elem_channels = [];      % array holding single element detector channels. two columns - column 1 = mda channel, column 2 = roi on single element detector list
    pilatus_channel = [];
    
    for idx = 1:numel(pt.detectors)
        if isempty(pilatus_channel)
            if strcmp(pt.detectors(idx).name, pilatus_name)
                pilatus_channel = idx;
            end
        end
        
        if contains(pt.detectors(idx).name, single_elem_detector_name)
            roi_num = strsplit(pt.detectors(idx).name, single_elem_detector_name);
            roi_num = str2double(roi_num{2});
            single_elem_channels = [single_elem_channels; idx, roi_num];
            
            % look for a user-supplied label for this ROI
            label_idx = find(roi_numbers == roi_num);
            if ~isempty(label_idx)
                label = roi_labels{label_idx};
            else
                label = 'Unlabeled';
            end
            
            single_elem_labels{size(single_elem_channels,1)} = label;
        end
        
    end
    
    sorted(1:numpts_m, 1:numpts_n) = struct(   'x', [],...
                                               'y', [],... 
                                               'diffraction', struct(       'twotheta', twoth,...
                                                                            'counts', zeros(1,num_bins)),...
                                               'fluorescence', struct(      'counts', [],...
                                                                            'label', '',...
                                                                            'channel', [],...
                                                                            'ROI', [])...
                                            );
                                        
    %% Pull diffraction and fluorescence data out of mda file
    summed_pilatus_ccd = zeros(size(qbin.twotheta));
    
    x_positions = dat.scan.sub_scans(1).positioners_data;
    x_positions = x_positions - min(x_positions);
    
    y_positions = dat.scan.positioners_data;
    y_positions = y_positions - min(y_positions);
    
    
    h_waitbar = waitbar(0,  'Processing mda file');
    for m = 1:numpts_m
        % update waitbar      
        if m > 1
            eta = toc*(numpts_m-m);
            waitbar(double(m)/double(numpts_m),  h_waitbar, sprintf('Processing mda file: ETA %.2f seconds', eta)); 
            tic;
        else
            waitbar(double(m)/double(numpts_m), h_waitbar, 'Processing mda file');
            tic;
        end
        
        
        scandat = dat.scan.sub_scans(m).detectors_data;
        
        for n = 1:numpts_n
            % Get realspace x and y coordinates
            sorted(m,n).x = x_positions(n);
            sorted(m,n).y = y_positions(m);
            
            % Get the pilatus ccd image and collapse it into twotheta
            % values
            ccdnum = scandat(n, pilatus_channel);
            filename = ['Images/' num2str(scannum) '/scan_' num2str(scannum) '_img_Pilatus_' num2str(ccdnum, '%6.6d') '.tif'];
            
            try
                ccd = double(imread(filename));
            catch
                ccd = zeros(size(qbin.twotheta)); % we dont read diffraction data if the image was not found
            end
            
            % CCD NOISE REMOVAL WOULD GO HERE %
            
            summed_pilatus_ccd = summed_pilatus_ccd + ccd;
            
            for twoth_idx = 1:num_bins
                pixels_in_range = abs(qbin.twotheta - twoth(twoth_idx)) < twoth_tolerance;
                sorted(m,n).diffraction.counts(twoth_idx) = sum(ccd(pixels_in_range));
            end
            
            % Get the single element fluorescence detector values here
            for fluo_idx = 1:size(single_elem_channels, 1)
                sorted(m,n).fluorescence(fluo_idx).counts = scandat(n, single_elem_channels(fluo_idx, 1));
                sorted(m,n).fluorescence(fluo_idx).label = single_elem_labels{fluo_idx}; 
                sorted(m,n).fluorescence(fluo_idx).channel = single_elem_channels(fluo_idx, 1);
                sorted(m,n).fluorescence(fluo_idx).ROI = single_elem_channels(fluo_idx, 2);
                
                
            end            
        end
    end
    
    close(h_waitbar);
    
    out.scan_number = scannum;
    out.data = sorted;
    out.summed_ccd = summed_pilatus_ccd;
end