function data_out=analyze_thscan_registered_interpolated(scannums,plotflag, ROIrange, XRFchan)

if(nargin<2) 
    plotflag = 0;
end
%% Edit detector channels here
if nargin < 4
    XRFchan = 39;   %set to Cr in four-element   39, 23
end
imagechan = 21; %Pilatus image index channel
thetachan = 52;
twothetachan = 51;
gammachan = 50;
rdetchan = 49;
normchan = 1;

%% Diffraction Pattern ROI on Pilatus CCD
    %CsPbBr Ortho Pilatus
  Xmax = 487;
  Ymax = 105;
    
%   ROIXstart = 431;
%   ROIXwidth = Xmax - ROIXstart;
%   ROIYstart = 1;
%   ROIYwidth = 195-ROIYstart;
if nargin > 2
    ROIXstart = ROIrange(1);
    ROIXwidth = ROIrange(2);
    ROIYstart = ROIrange(3);
    ROIYwidth = ROIrange(4);
else
    ROIXstart = 431;
    ROIXwidth = Xmax - ROIXstart;
    ROIYstart = 1;
    ROIYwidth = 195-ROIYstart;
end
%% Load all mda files, determine realspace range
data_out.thvals = zeros(max(size(scannums)),1);

% XRF1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],XRFchan,0,0);
% theta = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],thetachan,0,0);
% norm1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],normchan,0,0);
% twotheta = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],twothetachan,0,0);
% gamma = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],gammachan,0,0);
% rdet = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],rdetchan,0,0);

realx_min = inf;
realx_max = -inf;
realx_maxrange = 0;

disp('Loading Scan Data');
for mm = 1:length(scannums)
    
    %load scan data
    XRF{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],XRFchan,0,0);
    theta{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],thetachan,0,0);
    theta{mm} = theta{mm}(1,1,1);
    norm1{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],normchan,0,0);
    twotheta{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],twothetachan,0,0);
    gamma{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],gammachan,0,0);
    rdet{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],rdetchan,0,0);
    ccdnums{mm} = loadmda(['mda/26idbSOFT_' num2str(scannums(mm),'%4.4d') '.mda'],imagechan,0,0);

    %determine hybridx and realx range for each scan
    hybridx{mm} = XRF{mm}(1,:,3);
    realx{mm} = hybridx{mm}./sind(theta{mm});
    realx_norm{mm} = realx{mm} - min(realx{mm});

    if min(realx{mm}) < realx_min
        realx_min = min(realx{mm});
    end
    
    if max(realx{mm}) > realx_max
        realx_max = max(realx{mm});
    end   
    
    if max(realx_norm{mm}) > realx_maxrange
        realx_maxrange = max(realx_norm{mm});
    end
    
end

%% set interpolation range in realspace, create correlation map between interp realspace and scan hybridx points
interp_bins = 1e3;                                      %number of bins to divide first scan's hybridx range into.
% interp_x = linspace(realx_min, realx_max, interp_bins); %realspace interpolation range spanning range across all scans
interp_x = linspace(0, realx_maxrange, interp_bins);

corr_map = zeros(length(scannums), interp_bins);

disp('Interpolating and Registering XRF Maps');

for mm = 1:length(scannums)
    XRF_interp{mm} = zeros(size(XRF{1},1), interp_bins);        %XRF map, but on interpolated grid. will be used for alignment in realspace
    
    for ii = 1:interp_bins
        [~, corr_map(mm,ii)] = min(abs(interp_x(ii) - realx_norm{mm})); 
        XRF_interp{mm}(:,ii) = XRF{mm}(:, corr_map(mm,ii), 1);
    end
    
end

%% register XRF maps on interpolated grid
[transform, data_transformed, data, overlap] = imregister_interpolated(XRF_interp);

%% initialize pilatus ROI map for each pt in interpolated image grid
disp('Initializing Pilatus ROI Maps');
for ii =1:size(XRF{1},1)
    for jj=1:interp_bins
        data_out.ii(ii).jj(jj).im = zeros(ROIYwidth+1,ROIXwidth+1);
        data_out.ii(ii).jj(jj).immaxrc = zeros(ROIYwidth+1,ROIXwidth+1);
    end
end



chi2 = zeros(5,5);
I0 = sum(sum(norm1{1}(:,:,1)))/(size(norm1{1},1)*size(norm1{1},2)); %add normalization
XRF{1}(:,:,1) = I0*XRF{1}(:,:,1)./norm1{1}(:,:,1);
data_out.curve=zeros(2,max(size(scannums)));
data_out.Intensity = zeros(size(XRF{1}, 1), interp_bins);
data_out.XCent = zeros(size(XRF{1}, 1), interp_bins);
data_out.YCent = zeros(size(XRF{1}, 1), interp_bins);
data_out.thmax = zeros(size(XRF{1}, 1), interp_bins);
data_out.thcen = zeros(size(XRF{1}, 1), interp_bins);
data_out.ROIXstart = ROIXstart;
data_out.ROIYstart = ROIYstart;
data_out.ROIXwidth = ROIXwidth;
data_out.ROIYwidth = ROIYwidth;
data_out.twotheta = twotheta{1}(1,1,1);
data_out.gamma = gamma{1}(1,1,1);
data_out.rdet = rdet{1}(1,1,1);


%% Process rocking curves using registered maps
disp('Loading Pilatus Data');
h=waitbar(0,'Loading Pilatus Data');
% if(plotflag)
%     figure(101);
%     clf reset;
% %     size( XRF1 )
%     imagesc(XRF1(1,:,3),XRF1(:,1,2),XRF1(:,:,1));colormap inferno;axis square tight;
%     pause(0.5)
% end

for mm=1:length(scannums)
    waitbar((mm-1)/length(scannums));
    
    mdanum = scannums(mm);
    
    data_out.thvals(mm) = theta{mm};
    data_out.curve(1,mm) = theta{mm};
    data_out.scan(mm).scannum = scannums(mm);
    data_out.scan(mm).ccdnums = ccdnums{mm};
    data_out.scan(mm).XRF = XRF{mm};
    data_out.scan(mm).XRF_interp = XRF_interp{mm};
    data_out.scan(mm).RealXRange = realx_norm{mm};
    data_out.scan(mm).interpx = interp_x;
    data_out.scan(mm).imregtform = transform{mm};
    XRF{mm}(:,:,1) = I0*XRF{mm}(:,:,1)./norm1{mm}(:,:,1);
    
    %% collect Pilatus ROI to interpolated x pts
   
    
    for ii = 1:size(XRF{1},1)
        xrf_x_idx_prev = 0;
        waitbar((ii + (size(XRF{1},1)*(mm-1)))/(length(scannums)*size(XRF{1},1)));
        for jj= 1:interp_bins
            
            %Determine which point from original MDA file should correspond
            %to point on interpolated, translated maps. Points outside the
            %interpolation bins (ie, at the edges where translation pushes
            %the value out of interpolation range) are set to the edge
            %value. Edge of maps may be unreliable in output file
            
            translated_idx = jj + round(transform{mm}.T(3,1));
            if translated_idx > interp_bins
                translated_idx = interp_bins;
            elseif translated_idx <= 0
                translated_idx = 1;
            end
            
            xrf_x_idx = corr_map(mm, translated_idx);
            if xrf_x_idx == xrf_x_idx_prev
                data_out.ii(ii).jj(jj) = data_out.ii(ii).jj(jj-1);
            else
                xrf_x_idx_prev = xrf_x_idx;
                try
                    filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(data_out.scan(mm).ccdnums(ii,xrf_x_idx), '%6.6d') '.tif'];
                catch
                    fprintf( 'Error:\n' );
                    mm 
                    ii 
                    jj 
                    mdanum 
                    size( data_out.scan )
                    size( data_out.scan( mm ).ccdnums( ii, xrf_x_idx ) )
                end

                %ccd = double(imread(filename));

                % Modification by Siddharth Maddali, feed dummy zero array into
                % ccd variable in case of missing file.
                dummyArray = zeros( ROIYwidth+1, ROIXwidth+1 );
    %             [ ROIXwidth ROIYwidth ]
                try
                    ccd = double( imread(filename, 'PixelRegion', {[ROIYstart, ROIYstart+ROIYwidth],[ROIXstart, ROIXstart+ROIXwidth]}) );
                catch
                    ccd = dummyArray;
                end
    %             size( ccd )
                % End modification by Sid


    % %             Uncomment next line and comment out previous modification
    %               to revert from Sid's modifications and return to normal
    %               functioning.
                 %ccd = double(imread(filename, 'PixelRegion', {[ROIYstart, ROIYstart+ROIYwidth],[ROIXstart, ROIXstart+ROIXwidth]})); 
                %ccd1=ccd;
                ccd1=I0*ccd./norm1{mm}(ii,xrf_x_idx,1);
                ccd=ccd.*(ccd1>0);
                ccd(ccd<(2*std(ccd))) = 0;
                % hot pixel removal UNCOMMENT HERE
                %{
                ccd1 = zeros(size(ccd,1),size(ccd,2),4);
                ccd1(:,:,1) = circshift(ccd,[0 1]);
                ccd1(:,:,2) = circshift(ccd,[1 0]);
                ccd1(:,:,3) = circshift(ccd,[0 -1]);
                ccd1(:,:,4) = circshift(ccd,[-1 0]);
                ccd2 = median(ccd1,3);
                ccdmask = ccd>(ccd2+5);   %CHANGE HOT PIXEL THRESHOLD HERE
                ccd = ccd.*(1-ccdmask)+ccd2.*ccdmask;
                %}
       %%{         
                %ROI1 = ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth));
                ROI1 = ccd;

                %display(size(data_out.ii(ii).jj(jj).im));display(size(ROI1));
                data_out.ii(ii).jj(jj).im = data_out.ii(ii).jj(jj).im + ROI1;
                data_out.ii(ii).jj(jj).rc(mm) = sum(ROI1(:));


                %identify and save image from maximum of rocking curve at each
                %position
                if mm==1
                    data_out.ii(ii).jj(jj).immaxrc = ROI1;
                    data_out.ii(ii).jj(jj).th = theta{mm};
                elseif sum(ROI1(:))> sum(data_out.ii(ii).jj(jj).immaxrc(:)) %replace immaxrc image only if current ROI integration is greater than previous
                    data_out.ii(ii).jj(jj).immaxrc = ROI1;
                    data_out.ii(ii).jj(jj).th = theta{mm};
                end 

                %data_out.curve(2,mm) = data_out.curve(2,mm) +sum(sum(ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth))));
                data_out.curve(2,mm) = data_out.curve(2,mm) +sum(sum(ROI1));
            end
        end
    end
    %}
    if(plotflag)
    figure(101);
    clf reset;
    imagesc(XRF{1}(1,:,3),XRF{1}(:,1,2),data_out.scan(mm).XRF(:,:,1));colormap viridis;axis square tight;
    pause(0.5)
    end
end
close(h);

fluo = data_out.scan(1).XRF(:,:,1);
maxfluo = max(fluo(:));
cutoff = 0.3;
ind = find(fluo < cutoff*maxfluo);
%%{
disp('Calculating Centroids');
h=waitbar(0,'Calculating Centroids');
for ii =1:size(XRF{1},1)
    waitbar(ii/size(XRF{1},1));
    for jj=1:interp_bins
        for kk = 1:max(size(scannums))
            data_out.thcen(ii,jj) = data_out.thcen(ii,jj)+data_out.thvals(kk)*data_out.ii(ii).jj(jj).rc(kk);
        end
        data_out.thcen(ii,jj) = data_out.thcen(ii,jj)/sum(sum(data_out.ii(ii).jj(jj).rc));
        %ROI1 = data_out.ii(ii).jj(jj).immaxrc; %use to calculate pixel centroids from MAX
        ROI1 = data_out.ii(ii).jj(jj).im; %use to calculate pixel centroids from SUM
        data_out.Intensity(ii,jj)=sum(sum(ROI1));
        line1=sum(ROI1,1);  % horizontal
        line2=sum(ROI1,2);  % vertical
        data_out.thmax(ii,jj) = data_out.ii(ii).jj(jj).th;
        for kk=1:size(line1,2)
            data_out.XCent(ii,jj)=data_out.XCent(ii,jj)+kk*line1(kk)/data_out.Intensity(ii,jj);
        end
        data_out.XCent(ii,jj)=data_out.XCent(ii,jj)+ROIXstart;
        for kk=1:size(line2,1)
            data_out.YCent(ii,jj)=data_out.YCent(ii,jj)+kk*line2(kk)/data_out.Intensity(ii,jj);
        end
        data_out.YCent(ii,jj)=data_out.YCent(ii,jj)+ROIYstart;
    end
end
close(h);

%data_out.XCent(ind) = NaN;
%data_out.YCent(ind) = NaN;
%data_out.thmax(ind) = NaN;
%}
  if(plotflag)  
   figure(102);plot(data_out.curve(1,:),data_out.curve(2,:),'o','MarkerSize',10,'MarkerFaceColor','k');axis square;   
   figure(103);clf;imagesc(interp_x,XRF{1}(:,1,2),data_out.XCent);axis image;colormap hot; shading interp; title('ROI X Centroid');set(gca, 'YDir', 'normal');colorbar;
   figure(104);clf;imagesc(interp_x,XRF{1}(:,1,2),data_out.scan(1).XRF(:,:,1));axis image;colormap hot; shading interp; title('Cs XRF');set(gca, 'YDir', 'normal');colorbar;
   % Add CCD image numbers and pass structure to data cursor
   figure(105);clf;imagesc(interp_x,XRF{1}(:,1,2),data_out.YCent);axis image;colormap hot; shading interp; title('ROI Y Centroid');set(gca, 'YDir', 'normal');colorbar;
   %figure(106);clf;imagesc(XRF1(1,:,3),XRF1(:,1,2),data_out.thmax);axis image;colormap hot; shading interp; title('Maximum Theta');set(gca, 'YDir', 'normal');colorbar;
   figure(107);clf;imagesc(interp_x,XRF{1}(:,1,2),data_out.thcen);axis image;colormap hot; shading interp; title('Centroid Theta');set(gca, 'YDir', 'normal');colorbar;

  end
end




