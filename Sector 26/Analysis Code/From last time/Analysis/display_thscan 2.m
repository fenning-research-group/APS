function display_thscan(data)
XRF1 = data.scan(1).XRF;
for ii=1:max(size(data.scan))
    %figure(90+ii);clf
    figure(90);clf
    imagesc(XRF1(1,:,3),XRF1(:,1,2),data.scan(ii).XRF(:,:,1));axis square;colormap hot; shading interp; 
    title(['Ag XRF scan:' num2str(data.scan(ii).scannum)]);set(gca, 'YDir', 'normal');
    pause(0.5);
end
%%{
   figure(102);plot(data.curve(1,:),data.curve(2,:),'o','MarkerSize',10,'MarkerFaceColor','k');axis square;   
   figure(103);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data.XCent);axis square;colormap hot; shading interp; title('ROI X Centroid');set(gca, 'YDir', 'normal');colorbar;
%    figure(104);clf;hfig=imagesc(XRF1(1,:,3),XRF1(:,1,2),data.scan(1).XRF(:,:,1));axis square;colormap hot; shading interp; title('Ag XRF');set(gca, 'YDir', 'normal');colorbar;
   figure(104);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data.scan(1).XRF(:,:,1));axis square;colormap hot; shading interp; title('Au XRF');set(gca, 'YDir', 'normal');colorbar;
   % Add CCD image numbers and pass structure to data cursor
%             pass2click.data = data;
%             pass2click.xaxis = [min(XRF1(1,:,3)) max(XRF1(1,:,3)) size(XRF1(1,:,3),2)];
%             pass2click.yaxis = [min(XRF1(:,1,2)) max(XRF1(:,1,2)) size(XRF1(:,1,2),1)];
%             set(hfig, 'UserData', pass2click);
%             datacursormode on;
%             dcm_obj = datacursormode(gcf);
%             set(dcm_obj, 'DisplayStyle', 'window');
%            set(dcm_obj, 'UpdateFcn', @click4rock);
   figure(105);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data.YCent);axis square;colormap hot; shading interp; title('ROI Y Centroid');set(gca, 'YDir', 'normal');colorbar;
   %figure(106);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data.thmax);axis square;colormap hot; shading interp; title('Maximum Theta');set(gca, 'YDir', 'normal');colorbar;
   figure(107);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data.thcen);axis square;colormap hot; shading interp; title('Centroid Theta');set(gca, 'YDir', 'normal');colorbar;
   figure(108);clf;hfig=imagesc(XRF1(1,:,3),XRF1(:,1,2),data.Intensity);axis square;colormap hot; shading interp; title('Integrated Intensity');set(gca, 'YDir', 'normal');colorbar;
            pass2click.data = data;
            pass2click.xaxis = [min(XRF1(1,:,3)) max(XRF1(1,:,3)) size(XRF1(1,:,3),2)];
            pass2click.yaxis = [min(XRF1(:,1,2)) max(XRF1(:,1,2)) size(XRF1(:,1,2),1)];
            set(hfig, 'UserData', pass2click);
            datacursormode on;
            dcm_obj = datacursormode(gcf);
            set(dcm_obj, 'DisplayStyle', 'window');
           set(dcm_obj, 'UpdateFcn', @click4rock);

  % caxis([0 4000]);
%}
end