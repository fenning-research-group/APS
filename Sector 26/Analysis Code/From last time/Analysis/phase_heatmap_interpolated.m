function ROI_intensity=phase_heatmap_interpolated(scannums, qbin_data, plotflag)
%% User Inputs
    
    %material labels and diffraction peaks
    % CsPbI3
    materials = {'PbI2',...
                 'EuI',...
                 'EuCsPbI3'...
                };
    twotheta = {[23,12, 30.53, 35.17, 42.53, 37.07, 40.18, 34,41, 47.26, ]...
                [25.96, 32.74, 25.87, 38.44, 31.86, 27.24, 18.79, 30.75, 42.25, 46.97, 42.48, 21.61]...
                [14.1936, 20.1053, 28.4923, 38.0248]...
                };
%                 [14.0800   19.9600   24.5100   28.3800   31.8100   34.9400   40.5700   43.1500   45.6100   47.9700   50.2500   52.4600   54.6000   58.7200]...

            
            % MAPbI3
            
%                  'MAPbI3',...
%                 [25.51, 36.17, 28.59, 36.39, 28.31, 28.52, 51.75, 44.79, 38.47, 21.96, 38.21, 38.69, 46.92, 38.11, 40.87, 27.69, 31.32]...
    twotheta_tolerance = 0.2;
            
    % Edit detector channels here
    XRFchan = 39;   %set to Cs
    imagechan = 21; %Pilatus image index channel
    thetachan = 52;
    twothetachan = 51;
    gammachan = 50;
    rdetchan = 49;
    normchan = 1;
            


    if(nargin<2) 
        plotflag = 0;
    end

        
%% Code start

% Generate ROI matrices from twotheta values + qbin_data
    
    ROI = zeros(size(qbin_data.twotheta, 1), size(qbin_data.twotheta, 2), size(twotheta,2));
    for mat_idx = 1:size(twotheta,2)
        for th_idx = 1:numel(twotheta{mat_idx})
            ROI(:,:,mat_idx) = ROI(:,:,mat_idx) + (abs(qbin_data.twotheta - twotheta{mat_idx}(th_idx)) <= twotheta_tolerance);
        end
    end
    
    ROI = logical(ROI);
%APS Default code
    data_out.thvals = zeros(max(size(scannums)),1);

    XRF1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],XRFchan,0,0);
    norm1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],normchan,0,0);
    twotheta = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],twothetachan,0,0);
    gamma = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],gammachan,0,0);
    rdet = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],rdetchan,0,0);
%     for ii =1:size(XRF1,1)
%         for jj=1:size(XRF1,2)
%             data_out.ii(ii).jj(jj).im = zeros(ROIYwidth+1,ROIXwidth+1);
%             data_out.ii(ii).jj(jj).immaxrc = zeros(ROIYwidth+1,ROIXwidth+1);
%         end
%     end
%     chi2 = zeros(10,10);
    I0 = sum(sum(norm1(:,:,1)))/(size(norm1,1)*size(norm1,2)); %add normalization
%     XRF1(:,:,1) = I0*XRF1(:,:,1)./norm1(:,:,1);
%     data_out.curve=zeros(2,max(size(scannums)));
%     data_out.Intensity = zeros(size(XRF1,1),size(XRF1,2));
%     data_out.XCent = zeros(size(XRF1,1),size(XRF1,2));
%     data_out.YCent = zeros(size(XRF1,1),size(XRF1,2));
%     data_out.thmax = zeros(size(XRF1,1),size(XRF1,2));
%     data_out.thcen = zeros(size(XRF1,1),size(XRF1,2));
%     data_out.ROIXstart = ROIXstart;
%     data_out.ROIYstart = ROIYstart;
%     data_out.ROIXwidth = ROIXwidth;
%     data_out.ROIYwidth = ROIYwidth;
%     data_out.twotheta = twotheta(1,1,1);
%     data_out.gamma = gamma(1,1,1);
%     data_out.rdet = rdet(1,1,1);


%% Go through each scan
    h=waitbar(0,'Loading multiple scans');
    if(plotflag)
        figure(101);
        clf reset;
    %     size( XRF1 )
        imagesc(XRF1(1,:,3),XRF1(:,1,2),XRF1(:,:,1));colormap hot;axis square tight;
        pause(0.5)
    end

    for mm=1:max(size(scannums))
        waitbar(mm/max(size(scannums,2)), h);
        mdanum=scannums(mm);
        th_temp = loadmda(['mda/26idbSOFT_' num2str(mdanum,'%4.4d') '.mda'],thetachan,0,0);
        data_out.thvals(mm) = th_temp(1,1,1);
        data_out.curve(1,mm) = th_temp(1,1,1);
        data_out.scan(mm).scannum = mdanum;
        ccdnums = loadmda(['mda/26idbSOFT_' num2str(mdanum,'%4.4d') '.mda'],imagechan,0,0);
        XRF2 = loadmda(['mda/26idbSOFT_' num2str(mdanum,'%4.4d') '.mda'],XRFchan,0,0);
        norm2 = loadmda(['mda/26idbSOFT_' num2str(mdanum,'%4.4d') '.mda'],normchan,0,0);
        
        data_out.scan(mm).ccdnums = ccdnums;
%         XRF2(:,:,1) = I0*XRF2(:,:,1)./norm2(:,:,1);
        %{
        % lineup with fluorescence centroid
        y_mass = sum(XRF2(:,:,1),1);
        x_mass = sum(XRF2(:,:,1),2);
        yline = ([1:1:numel(y_mass)]);
        xline = transpose([1:1:numel(x_mass)]);
        shiftx = round(round(numel(xline)/2) - sum((x_mass).*xline)/sum((x_mass)));
        shifty = round(round(numel(yline)/2) - sum((y_mass).*yline)/sum((y_mass)));
        data_out.scan(mm).XRF = XRF2;  %keep the same scanning axes
        data_out.scan(mm).XRF(:,:,1) = circshift(XRF2(:,:,1),[shiftx,shifty]);
        data_out.scan(mm).ccdnums = circshift(ccdnums,[shiftx,shifty]);
        %}
       %%{
        % line up with difference minimization - make sure chi2 size much less
        % than either scanning axis
%         if(mm>1)
%         trim1 = round(size(chi2,1)/2);
%         trim2 = round(size(chi2,2)/2);
%         range1 = trim1:(size(XRF1,1)-trim1);
%         range2 = trim2:(size(XRF1,2)-trim2);
%          for jjj = 1:size(chi2,1)
%             for kkk = 1:size(chi2,2)
%                 %chi2(jj,kk) = sum(sum((XRF1(:,:,1) - circshift(XRF2(:,:,1),[jj-10,kk-10])).^2));
%                 chi2(jjj,kkk) = sum(sum((XRF1(range1,range2,1) - circshift(XRF2(range1,range2,1),[jjj-round(size(chi2,1)/2),kkk-round(size(chi2,2)/2)])).^2));
%             end
%         end
%         [xcen1,ycen1,intemp] = find(chi2==min(min(chi2)));
%         %figure;imagesc(chi2)
%         %display(xcen1-round(size(chi2,1)/2));display(ycen1-round(size(chi2,2)/2))
%         %display([xcen,ycen,xcen-round(size(chi2,1)/2),ycen-round(size(chi2,2)/2)])
% %         data_out.scan(mm).XRF = circshift(XRF2,[xcen1-round(size(chi2,1)/2),ycen1-round(size(chi2,2)/2)]);
% %         data_out.scan(mm).ccdnums = circshift(ccdnums,[xcen1-round(size(chi2,1)/2),ycen1-round(size(chi2,2)/2)]);
%         else
%             data_out.scan(mm).XRF=XRF2;
%             data_out.scan(mm).ccdnums = ccdnums;
%         end

        ROI_intensity = zeros(size(XRF1,1), size(XRF1,2), size(ROI,3));

        for ii = 1:size(XRF1,1)
            for jj= 1:size(XRF1,2)
                try
                    filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(data_out.scan(mm).ccdnums(ii,jj), '%6.6d') '.tif'];
                catch
                    fprintf( 'Error:\n' );
                    mm 
                    ii 
                    jj 
                    mdanum 
                    size( data_out.scan )
                    size( data_out.scan( mm ).ccdnums( ii, jj ) )
                end

                %ccd = double(imread(filename));

                % Modification by Siddharth Maddali, feed dummy zero array into
                % ccd variable in case of missing file.
                dummyArray = zeros(size(qbin_data.twotheta));

                try
                    ccd = double( imread(filename) );
                catch
                    ccd = dummyArray;
                end

                ccd1=I0*ccd./norm2(ii,jj,1);
                ccd=ccd.*(ccd1>0);

                for kk = 1:size(ROI, 3)

            % %             Uncomment next line and comment out previous modification
            %               to revert from Sid's modifications and return to normal
            %               functioning.
                         %ccd = double(imread(filename, 'PixelRegion', {[ROIYstart, ROIYstart+ROIYwidth],[ROIXstart, ROIXstart+ROIXwidth]})); 
                        %ccd1=ccd;


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
                        ROI_intensity(ii,jj,kk) = ROI_intensity(ii,jj,kk) + sum(sum(ccd(ROI(:,:,kk))));    %intensity of ccd in ROI region

                        %identify and save image from maximum of rocking curve at each
                        %position
                        %data_out.curve(2,mm) = data_out.curve(2,mm) +sum(sum(ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth))));

                end
            end
        %}
        end
    end
close(h);

%% Plot Results
    if plotflag
        plotphasedat(ROI_intensity, materials);
    end
        
end



