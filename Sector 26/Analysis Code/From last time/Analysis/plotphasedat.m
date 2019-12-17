function plotphasedat(raw_data, fpath)
    %% Inputs
    n = 2;      %number of standard deviations around mean to scale image by
    
    
    %% Code Start
    
    if nargin > 1
        writefile = true;
    else
        writefile = false;
    end
    %% Plot phase maps (diffracted intensity)
    
    num_phases = numel(raw_data.materials);
    hfig_diff = figure;
    hold on;
    
    %subplot ratio. max 3 columns
    xmax = 3;
    y = ceil(num_phases/xmax);
    x = min(xmax, num_phases);
    
    for i = 1:num_phases
        phasedat = raw_data.materials(i).diffractionmap;
        meanval(i) = mean(phasedat(:));
        stdval(i) = std(phasedat(:));
        figure(hfig_diff);
        subplot(y,x, i);
%         map = cat(3, ones(size(phasedat(:,:,1)))*colors(i,1), ones(size(phasedat(:,:,1)))*colors(i,2), ones(size(phasedat(:,:,1)))*colors(i,3));
        h = imagesc(phasedat);
        h.Parent.CLim = [meanval(i)-n*stdval(i) meanval(i)+n*stdval(i)];
%         set(h, 'AlphaData', phasedat_scaled(:,:,i));
        title(raw_data.materials(i).name);
        colormap(viridis);
        pbaspect([1 1 1]);
        hfig_temp = figure;
        h = imagesc(phasedat);
        h.Parent.CLim = [meanval(i)-n*stdval(i) meanval(i)+n*stdval(i)];
%         set(h, 'AlphaData', phasedat_scaled(:,:,i));
        title([raw_data.materials(i).name ' Diffraction Map']);
        colormap(viridis);
        pbaspect([1 1 1]);
        
        if writefile
            colorbar;
            prettyplot;
            export_fig(fullfile(fpath, ['_' raw_data.materials(i).name '_diffractionmap']), '-m1', '-painters');
            close(hfig_temp);
        end
    end
    figure(hfig_diff);
    [~, tempstr] = fileparts(fpath);
    sgtitle([tempstr ' Diffraction Maps']);
    export_fig(fullfile(fpath, ['_all_diffractionmap']), '-m3', '-painters');
    close(hfig_diff);
    
    %% Plot elemental maps (XRF signal)
    
    num_elems = numel(raw_data.xrfmaps);
    hfig_xrf = figure;
    hold on;
    
        
    %subplot ratio. max 3 columns
    xmax = 3;
    y = ceil(num_elems/xmax);
    x = min(xmax, num_elems);
    
    for i = 1:num_elems
        phasedat = raw_data.xrfmaps(i).map;
        meanval(i) = mean(phasedat(:));
        stdval(i) = std(phasedat(:));
        figure(hfig_xrf);
        subplot(y,x,i);
%         map = cat(3, ones(size(phasedat(:,:,1)))*colors(i,1), ones(size(phasedat(:,:,1)))*colors(i,2), ones(size(phasedat(:,:,1)))*colors(i,3));
        h = imagesc(phasedat);
        h.Parent.CLim = [meanval(i)-n*stdval(i) meanval(i)+n*stdval(i)];
%         set(h, 'AlphaData', phasedat_scaled(:,:,i));
        title(raw_data.xrfmaps(i).name);
        colormap(inferno);
        pbaspect([1 1 1]);
        
        hfig_temp = figure;
        h = imagesc(phasedat);
        h.Parent.CLim = [meanval(i)-n*stdval(i) meanval(i)+n*stdval(i)];
%         set(h, 'AlphaData', phasedat_scaled(:,:,i));
        title([raw_data.xrfmaps(i).name ' XRF Map']);
        colormap(inferno);
        pbaspect([1 1 1]);
        
        if writefile
            colorbar;
            prettyplot;
            export_fig(fullfile(fpath, ['_' raw_data.xrfmaps(i).name '_xrfmap']), '-m1', '-painters');
            close(hfig_temp);
        end
    end
    figure(hfig_xrf);
    sgtitle([tempstr ' Elemental Maps']);
    export_fig(fullfile(fpath, ['_all_xrfmap']), '-m3', '-painters');
    close(hfig_xrf);    
end