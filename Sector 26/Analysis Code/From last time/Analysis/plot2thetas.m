function plot2thetas(qbin_dat, phasemap_output)
%% User Inputs
    opacity = 0.4;
    twotheta_tolerance = 0.05;
    
%% Code Start

    materials = phasemap_output.materials;
    totaldiff = phasemap_output.summed_diffraction;
    
    meandiff = mean(mean(totaldiff));
    stddiff = std(std(double(totaldiff)));
    
%     totaldiff_norm = (totaldiff - min(totaldiff(:))) / (max(totaldiff(:)) - min(totaldiff(:)));
%     if nargin < 4       
%         plotfig = openfig('./blankpowderdiff.fig');
%     else
        plotfig = figure;
        
        him = imagesc(totaldiff);
        him.Parent.CLim = [meandiff - 3*stddiff meandiff + 3*stddiff];
        colormap(gray);
        hold on;
%     end
    
    
    
    twotheta_map = qbin_dat.twotheta;
    colors = linspecer(numel(materials));
    
    for matl_idx = 1:numel(materials)
        inrange = zeros(size(totaldiff(:,:,1)));
        im_temp = getimage(plotfig);
        for i = 1:numel(materials(matl_idx).twothetas)
            inrange = inrange + (abs(twotheta_map - materials(matl_idx).twothetas(i))<=twotheta_tolerance);
%             disp(materials(matl_idx).twothetas(i));
%             disp(sum(inrange(:)));
        end
        
        inrange = logical(inrange);
        
%         for m = 1:size(im_temp,1)
%             for n = 1:size(im_temp,2)
%                 if inrange(m,n)
%                     im_temp(m,n,:) = colors(matl_idx, :);
%                 end
%             end
%         end
        colorlayer = cat(3, ones(size(im_temp(:,:,1)))*colors(matl_idx,1), ones(size(im_temp(:,:,1)))*colors(matl_idx,2), ones(size(im_temp(:,:,1)))*colors(matl_idx,3));
        h = imshow(colorlayer);
        set(h, 'AlphaData', inrange*opacity);        
    end
    
    for matl_idx = 1:numel(materials)
        text(size(twotheta_map,2)*0.01,size(twotheta_map,1)*0.1*matl_idx,  materials(matl_idx).name, 'color', colors(matl_idx, :));
    end
    
end