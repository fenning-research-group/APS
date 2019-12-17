function show_sum_thetascan_rois(summed_thetascan_rois,mdanum)
    ROIs = [456, 17, 117, 15; 458, 15, 95, 114-95; 352, 13, 27, 13; 460, 14, 76, 13; 461, 12, 78, 8];

    XRFchan = 39;
    XRF = loadmda(['mda/26idbSOFT_' num2str(mdanum, '%4.4d') '.mda'],XRFchan,0,0);
    XRF_norm = log(XRF(:,:,1))/max(max(log(XRF(:,:,1))));
    domainsfig = figure;
    imshow(XRF_norm);
    hold on;  
    num_scans = size(summed_thetascan_rois,3);
    colors = jet(num_scans);
    
    for i = 1:num_scans
        color_filter = cat(3,ones(size(summed_thetascan_rois(:,:,i)))*colors(i,1), ones(size(summed_thetascan_rois(:,:,i)))*colors(i,2), ones(size(summed_thetascan_rois(:,:,i)))*colors(i,3));
        scan_norm = summed_thetascan_rois(:,:,i)/max(max(summed_thetascan_rois(:,:,i)));
        threshold = 0.1;
        scan_norm(scan_norm<threshold) = 0;
        scan_norm(scan_norm>0) = 1;
        h = imshow(color_filter);
        set(h, 'AlphaData', scan_norm);
    end
    xrdfig = openfig('/CNMshare/savedata/2018R3/20181023/Analysis/CsPbBr3_ortho_ccd.fig');
    hold on;
    for kk =1:num_scans
        rectangle('Position', [ROIs(kk,1) ROIs(kk,3) ROIs(kk,4) ROIs(kk,2)], 'EdgeColor', colors(kk,:), 'LineWidth', 2);
    end
end