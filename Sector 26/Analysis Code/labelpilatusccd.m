function labelpilatusccd(ccd, qbin, peaks)
   %% inputs
   twoth_tolerance = 0.1;

   %% generate thresholded ccd image and find centroids
   ccd_bw = imbinarize(mat2gray(ccd), 'global');
   ccd_bw = imfill(ccd_bw, 'holes');
   
   Ilabel = bwlabel(ccd_bw);
   stats = regionprops(Ilabel, 'centroid');
   
   %% display ccd image
   figure, hold on;
   h = imagesc(ccd);
   colormap(h.Parent, gray);
   set(h.Parent, 'YDir', 'reverse');
   xlim([0 size(ccd, 2)]);
   ylim([0 size(ccd, 1)]);
   
   all_twotheta = [];
   for idx = 1:numel(stats)
        y = round(stats(idx).Centroid(1));
        x = round(stats(idx).Centroid(2));
        twotheta = round(qbin.twotheta(x,y), 2);
        if ~any(abs(all_twotheta-twotheta) < twoth_tolerance)
            all_twotheta = [all_twotheta; twotheta];
            plot(stats(idx).Centroid(1), stats(idx).Centroid(2), 'rx', 'markersize', 10);
            text(stats(idx).Centroid(1)+10, stats(idx).Centroid(2)-5, num2str(twotheta), 'color', [1 0 0]);
        end
   end    
end