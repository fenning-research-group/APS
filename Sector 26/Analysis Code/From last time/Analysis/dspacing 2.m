function dataout = dspacing(data,data_q,plotflag)
% function for interpolating d spacing from a given xcentroid / ycentroid
% this needs analyze_thscan to be run to create data structure
%load('powder_fit_v1.mat');dspac1 = dspacing(data_roxx,data_q.twotheta,1);
% to export for origin in csv >>dlmwrite('strain_leaf2.csv',dspac3(:,:,2));

Ekev = 12.0; % beam energy in kev
kb = 2*pi*Ekev/12.39842; % beam momentum 1/A
lambda = 2*pi/kb;
dataout = zeros(size(data.XCent,1),size(data.XCent,2),4);
twothmat = data_q.twotheta;
gammamat = data_q.gamma;

XRF1 = data.scan(1).XRF;
maxfluo = max(max(XRF1(:,:,1)));
cutoff = 0.3;
mask = XRF1(:,:,1) > (cutoff*maxfluo);

for ii = 1:size(data.XCent,1)
    for jj = 1:size(data.XCent,2)
        dataout(ii,jj,1) = interp2(twothmat,data.XCent(ii,jj),data.YCent(ii,jj));
        dataout(ii,jj,3) = interp2(gammamat,data.XCent(ii,jj),data.YCent(ii,jj));
    end
end

dataout(:,:,2) = sqrt(3)*lambda./(2*sind(dataout(:,:,1)/2));

imtot = zeros(size(data.ii(1).jj(1).im));
for ii =1:size(XRF1,1)
    for jj=1:size(XRF1,2)
        imtot = imtot + data.ii(ii).jj(jj).im;
        dataout(ii,jj,4) = sum(sum(data.ii(ii).jj(jj).im));
    end
end

if(plotflag)
    figure(301);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),dataout(:,:,1));
    axis image;  set(gca, 'YDir', 'normal');colormap parula;colorbar;title('Interpolated Two theta');
    figure(302);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),dataout(:,:,2));
    axis image; set(gca, 'YDir', 'normal');colormap parula;colorbar;title('Lattice constant');
    figure(303);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),100*(dataout(:,:,2)-4.0788)/4.0788);
    axis image; set(gca, 'YDir', 'normal');colormap parula;colorbar;title('Percent strain');
        figure(304);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),data.thcen);
    axis image; set(gca, 'YDir', 'normal');colormap parula;colorbar;title('Interpolated theta');
%     figure(304);
%     tempim = mask.*(100*(dataout(:,:,2)-4.0788)/4.0788);
%     imagesc(XRF1(1,:,3)./sind(12.5),XRF1(:,1,2),tempim);axis image;colormap parula;colorbar;title('Percent strain');
     figure(305);
     imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),XRF1(:,:,1));
     axis image; set(gca, 'YDir', 'normal');colormap hot;colorbar;title('Au XRF');
%      figure(306);
%      tempim = mask.*(100*(dataout(:,:,2)-4.0788)/4.0788);
%      imagesc(XRF1(1,:,3)./sind(12.5),XRF1(:,1,2),tempim);
%      axis image;colormap parula;colorbar;title('Percent strain');
%    figure(307);
%     imagesc(XRF1(1,:,3),XRF1(:,1,2),XRF1(:,:,1));
%     axis image;colormap hot;colorbar;title('Au XRF');
%     XRFgray = (XRF1(:,:,1)-min(min(XRF1(:,:,1))))./(max(max(XRF1(:,:,1)))-min(min(XRF1(:,:,1))));
%     XRDgray = (dataout(:,:,2)-min(min(dataout(:,:,2))))./(max(max(dataout(:,:,2)))-min(min(dataout(:,:,2))));
%     figure(308);im1(:,:,1)=XRFgray;im1(:,:,2)=XRFgray;im1(:,:,3)=XRDgray;
%     image(XRF1(1,:,3),XRF1(:,1,2),im1);
%     axis image;colorbar;title('Overlay 1');    
%     figure(309);im1(:,:,1)=XRFgray;im1(:,:,2)=XRDgray;im1(:,:,3)=XRFgray;
%     image(XRF1(1,:,3),XRF1(:,1,2),im1);
%     axis image;colorbar;title('Overlay 2');
%     figure(310);im1(:,:,1)=XRDgray;im1(:,:,2)=XRFgray;im1(:,:,3)=XRFgray;
%     image(XRF1(1,:,3),XRF1(:,1,2),im1);
%     axis image;colorbar;title('Overlay 3');
%     figure(311);im1(:,:,1)=XRDgray;im1(:,:,2)=XRDgray;im1(:,:,3)=zeros(size(XRDgray));
%     image(XRF1(1,:,3),XRF1(:,1,2),im1,'AlphaData',1-XRFgray);
%     axis image;colorbar;title('Overlay 4');
    figure(312);
    imagesc(log(imtot+1));colormap hot; axis square
        figure(313);
    imagesc(imtot);colormap hot; axis square
        figure(314);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),dataout(:,:,3));
    axis image; set(gca, 'YDir', 'normal');colormap parula;colorbar;title('Interpolated Gamma');
            figure(315);
    imagesc(XRF1(1,:,3)./sind(13.5),XRF1(:,1,2),dataout(:,:,4));
    axis image; set(gca, 'YDir', 'normal'); colormap parula;colorbar;title('Integrated Diffraction Intensity');
    figure(316);clf;yyaxis left;
    plot(XRF1(:,1,2),100*(dataout(:,12,2)-dataout(2,12,2))./dataout(2,12,2),'LineWidth',2,'color','blue');
    ylabel('Percent relative strain (dc/c)','color','blue')
    hold on;
    yyaxis right;
    plot(XRF1(:,1,2),dataout(:,12,3),'LineWidth',2,'color','red');
    ylabel('Lattice tilt (deg)');title('Strain and Tilt Across GB')
    
end

end