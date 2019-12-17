function plot26(simple26, xrfchannels)
    dat = simple26.data;  
    
    numpts_m = size(dat, 1);
    numpts_n = size(dat,2);
    all_ROI = vertcat(dat(1,1).fluorescence.ROI);
    
    
%     colormap_titles = {'Blues', 'Oranges', 'Greens', 'Reds'};
%     colormaps = zeros(256,3, numel(colormap_titles));
%     for idx = 1:numel(colormap_titles)
%         colormaps(:,:,idx) = cbrewer('seq', colormap_titles{idx}, 256);
%     end
    
    
    if nargin < 2
        xrfchannels = all_ROI;
    end
       
    xrf_idx = zeros(size(xrfchannels));
    
    for idx = 1:numel(xrf_idx)
        xrf_idx(idx) = find(all_ROI == xrfchannels(idx));
        
        if isempty(xrf_idx(idx))
            errorstring = 'Invalid XRF ROIs supplied - valid ROIs are: ';
            for m = 1:numel(all_ROI)
                errorstring = [errorstring num2str(all_ROI(m)) ', '];
            end
            errorstring(end-2:end) = [];
            
            error(errorstring);
        end
        
        xrf_label{idx} = dat(1,1).fluorescence(idx).label;
    end
%         
%     if isempty(xrf_idx)
%         xrf_idx = 1;
%         disp('XRF index not found: defaulting to first index available');
%     end
    
    intensity = zeros(size(dat,1), size(dat,2), numel(xrfchannels));

    for m = 1:numpts_m
        for n = 1:numpts_n
            for k = 1:numel(xrfchannels)
                intensity(m,n,k) = dat(m,n).fluorescence(xrf_idx(k)).counts;
            end
        end
    end
    
    %% plot Results
    
    %prepare colorscales for each xrf channel
%     colormaps = repmat(linspace(0,1,256)', 1, 3, numel(xrf_idx));
    colors = linspecer(numel(xrf_idx));
    
    hfig = figure;
    hold on;
%     opacity = 1/numel(xrf_idx);
    opacity = 1;
    
    for idx = 1:numel(xrf_idx)
        this_scan = intensity(:,:,idx);
        scan_norm = (this_scan - min(this_scan(:))) / (max(this_scan(:)) - min(this_scan(:)));
        
%         if idx > 1
%             axes();
%         end
        
%         h = imagesc(1:numpts_m, 1:numpts_n, ones(numpts_m, numpts_n));
%         colormap(h.Parent, colors(idx,:));
%         h.AlphaData = scan_norm*opacity;
        
        for layeridx = 1:3
            colorlayer(:,:,layeridx) = ones(numpts_m, numpts_n)*colors(idx,layeridx);
        end

        h = imshow(colorlayer, 'InitialMagnification', 'fit');
        h.AlphaData = scan_norm*opacity;
%         xlim([0 numpts_n]);
%         ylim([0 numpts_m]);

%         h = imagesc(1:numpts_m, 1:numpts_n, intensity(:,:,idx));
%         colormap(h.Parent, colormaps(:,:,idx));
%         h.AlphaData = opacity;
    end
    
    %add text over plot
    
    y_max = max(ylim);
    y_range = y_max - min(ylim);
    
    x_min = min(xlim);
    x_range = max(xlim) - x_min;
    
    for idx = 1:numel(xrf_idx)
        text(x_min + 0.05*x_range, y_max - 0.05*y_range*idx, xrf_label{idx}, 'Color', colors(idx, :));
    end
    
    %% Interactive tooltip to display scan on plot
    
    %data to be passed to click function that generates plots
    clickdata.data = dat;
    
    set(hfig, 'UserData', clickdata);
    
    hclick = datacursormode(hfig);
    set(hclick, 'DisplayStyle', 'Window',...
                'Enable', 'on',...
                'UpdateFcn', @click4pattern);
end


function txt = click4pattern(~, event_obj)
    position_tolerance = 0.0001;        % used to avoid rounding errors from preventing scan index lookup by comparing click position to data
    
    position = get(event_obj, 'Position');
    hfig =get(event_obj, 'Target');
    data = hfig.Parent.Parent.UserData.data;
    
%     xidx = find(abs(data.x_grid(1,:) - position(1)) < position_tolerance);
%     yidx = find(abs(data.y_grid(:,1) - position(2)) < position_tolerance);
    
    xidx = round(position(1));
    yidx = round(position(2));

    figure(200);
    twotheta = data(yidx, xidx).diffraction.twotheta;
    counts = data(yidx, xidx).diffraction.counts;
    
    plot(twotheta, counts);
%     plot(wl, spec);
    title(['X: ' num2str(position(1)) ' Y: ' num2str(position(2))]);
    xlabel('2\Theta (9 keV)');
    ylabel('Counts');
    xlim([min(twotheta) max(twotheta)]);
%     prettyplot('colorful');
    
    txt = { 'none',...
            'none',...
            'none'...
          };
      
    figure(hfig.Parent.Parent);
end