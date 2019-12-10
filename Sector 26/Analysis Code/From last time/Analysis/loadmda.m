function [dataout] = loadmda(filename, detector, print2screen, plotflag, analyzeflag, expandedOutputflag)

% data = loadmda(filename, detector, print2screen, plotflag, analyzeflag, expandedOutputflag)
%
% Stephan Hruszkewycz 1-12-09
% EDITED to use mdaload Martin Holt 1-19-16
% EDITED for expanded output using structured data Chris Richardson 2-26-16
%
% This function serves to extract useable matrices and visually represent
% data taken from the nested data structure created by the mdaload.m function
%
%
% Input variables for loadmda:
%
%   filename - Matlab string containing the name of the file.  The name
%               must contain the entire path of the file if working outside
%               the data folder.
%   detector - loadmda extracts ONE detector channel every time it is
%               invoked.  The 'detector' input variable should be the
%               process variable channel number (3 for detector D03, etc)
%   print2screen - 'y' or 'n' flag that determines whether a summary is
%               is displayed in Matlab. Suppressing output is useful when
%               loading large batches of .mda files.
%   plotflag - 'y' or 'n' flag that determines whether loaded data will be
%               automatically plotted in a Matlab figure.  Area scans are
%               plotted using 'surf' and can be rotated to show topology.
%               Scans acquiring diffraction or fluorescence will have the Data
%               Cursor icon of the figure window modified to interactively
%               display CCD data or MCA data (or both).
%
%   analyzeflag - perform analysis of a region of interest (ROI) using 
%               background substracted data.  The X-centroid, Y-centroid
%               intensity, are shown in individual windows.
%
%   expandedOutputflag - Save dataout in an alternative format using cells
%               such that the analyzed data is all available for further 
%               post-processing
%
% Output variables:
%
% 	data - This is a Matlab matrix containing the positioner and detector
%           data from the .mda file.  Line scans will return a two-
%           columned Matlab matrix where data(:,1) is the positioner and
%           data(:,2) is the detector readout.  Area scans are returned as
%           a 3D Matlab matrix where data(:,:,1) is the detector
%           readout, data(:,:,2) is the corresponding values of the outer
%           loop positioner, and data(:,:,3) contains the positions of the
%           inner loop positioner.
%
% WARNING:  THIS FUNCTION REQUIRES MDALOAD.M IN ORDER TO WORK
% For more info on mdaload.m structure, see pdf document mda-load.pdf.

fid = fopen(filename,'r','b');

if(fid==-1)
    error('The MDA file specified does not exist');
end

if(nargin<2)
    detector=1;
    print2screen='y';
    plotflag='y';
    analyzeflag = 'n';
    expandedOutputflag = 'n';
end

if(nargin<3)
    print2screen='y';
    plotflag='y';
    analyzeflag = 'n';
    expandedOutputflag = 'n';
end

if(nargin<4)
    plotflag='y';
    analyzeflag = 'n';
    expandedOutputflag = 'n';
end

if(nargin<5)
    analyzeflag = 'n';
    expandedOutputflag = 'n';
end

if(nargin<6)
    expandedOutputflag = 'n';
end

if((plotflag=='y')||(plotflag==1)); plotflag = 1;
else plotflag = 0;end

if((print2screen=='y')||(print2screen==1)); print2screen = 1;
else print2screen=0; end

if((analyzeflag=='y')||(analyzeflag==1)); analyzeflag = 1;
else analyzeflag=0; end

if((expandedOutputflag=='y')||(expandedOutputflag==1)); expandedOutputflag = 1;
else expandedOutputflag=0; end

if(fid>0)
    fclose(fid); %it will be re-opened by mdaload
    
    mda = mdaload(filename);
    
    if(analyzeflag)
        % Load the appropriate background image to subtract from the image
        % data files
        clear ccd2;
        %ccd2=double(imread('Images/22/scan_22_img_000392.tif')); % 0.7 second dark
        %ccd2 = double(imread('Images/image_014554.tif')); % 4 second dark
        %ccd2 = double(imread('Images/image_007919.tif')); % 8 second dark        
        %ccd2 = double(imread('Images/image_042117.tif')); % 16 second dark
        %ccd2 = double(imread('Images/image_007909.tif')); % 30 second dark
        %ccd2 = double(imread('Images/image_007889.tif')); % 60 second dark
        
        % If an ROI is not defined becuase all of the ROI settings are 
        % commented, the ROI is graphically created from the user`
        % later on in this script
        
        % Uncomment the following sections to select a predetermined ROI


%          Scan 14 rocking curve analysis both tilt peaks in ROI
          %ROIXstart = 180;
          %ROIXwidth = 200;
          %ROIYstart = 50;
          %ROIYwidth = 400;
%          Scan 123 / 124 comparison analysis of centroids
%          ROIXstart = 260;
%          ROIXwidth = 200;
%          ROIYstart = 1;
%          ROIYwidth = 256;
%          Scan 46 / 47 comparison analysis of centroids
%          ROIXstart = 180;
%          ROIXwidth = 250;
%          ROIYstart = 5;
 %         ROIYwidth = 300;
%          Scan 229 minus 225 comparison analysis of centroids
            % Previously
          ROIXstart = 413;
          ROIXwidth = 443-414;
          ROIYstart = 137;
          ROIYwidth = 164-137;
%             ROIXstart = 243;
%             ROIXwidth = 180;
%             ROIYstart = 100;
%             ROIYwidth = 214;

        
    end
    
    % Determine scan dimensions, check for detector channel and ccd image channel
    
    scan_dim = mda.scan.scan_rank;
    
    xrfscan=0;imagechan=0;detchan=0;
    
    if(scan_dim > 3);error('Higher dimensional scan loading not supported');end
    
    if(scan_dim == 3)
        if(max(size(mda.scan.sub_scans(1).sub_scans(1).detectors_data))>1023)
            xrfscan = 1;
            for ii=1:max(size(mda.scan.sub_scans(1).detectors))
                if(isempty(strfind(mda.scan.sub_scans(1).detectors(ii).name,'cam1:FileNumber'))==0)
                    imagechan=ii;
                end
                if(mda.scan.sub_scans(1).detectors(ii).number==detector-1)
                    detchan=ii;
                end
            end
            if(detchan==0)
                for ii=1:max(size(mda.scan.sub_scans(1).detectors))
                    fprintf([num2str(mda.scan.sub_scans(1).detectors(ii).number+1) ': ' ...
                        mda.scan.sub_scans(1).detectors(ii).name '\n']);
                end
                error('Detector NOT FOUND: check above list');
            end
        else
            error('Higher dimensional scan loading not supported');
        end
    end
        
    if(scan_dim == 2)
        if(max(size(mda.scan.sub_scans(1).detectors_data))>1023)
            xrfscan = 1;
            for ii=1:max(size(mda.scan.detectors))
                if(isempty(strfind(mda.scan.detectors(ii).name,'cam1:FileNumber'))==0)
                    imagechan=ii;
                end
                if(mda.scan.detectors(ii).number==detector-1)
                    detchan=ii;
                end
                if(isempty(strfind(mda.scan.detectors(ii).name,'m34.RBV'))==0)
%                if(isempty(strfind(mda.scan.detectors(ii).name,'Ave1'))==0)
                    xposchan=ii;
                end
                if(isempty(strfind(mda.scan.detectors(ii).name,'m35.RBV'))==0)
%                if(isempty(strfind(mda.scan.detectors(ii).name,'Ave2'))==0)
                    yposchan=ii;
                end
                

            end
            if(detchan==0)
                for ii=1:max(size(mda.scan.detectors))
                    fprintf([num2str(mda.scan.detectors(ii).number+1) ': ' ...
                        mda.scan.detectors(ii).name '\n']);
                end
                error('Detector NOT FOUND: check above list');
            end
        else
            for ii=1:max(size(mda.scan.sub_scans(1).detectors))
                if(isempty(strfind(mda.scan.sub_scans(1).detectors(ii).name,'cam1:FileNumber'))==0)
                    imagechan=ii;
                end
                if(mda.scan.sub_scans(1).detectors(ii).number==detector-1)
                    detchan=ii;
                end
            end
            if(detchan==0)
                for ii=1:max(size(mda.scan.sub_scans(1).detectors))
                    fprintf([num2str(mda.scan.sub_scans(1).detectors(ii).number+1) ': ' ...
                        mda.scan.sub_scans(1).detectors(ii).name '\n']);
                end
                error('Detector NOT FOUND: check above list');
            end
        end
    end
    
    if(scan_dim == 1)
        for ii=1:max(size(mda.scan.detectors))
            if(isempty(strfind(mda.scan.detectors(ii).name,'cam1:FileNumber'))==0)
                imagechan=ii;
            end
            if(isempty(strfind(mda.scan.detectors(ii).name,'m34.RBV'))==0)
                xposchan=ii;
            end
            if(isempty(strfind(mda.scan.detectors(ii).name,'m35.RBV'))==0)
                yposchan=ii;
            end
            if(mda.scan.detectors(ii).number==detector-1)
                detchan=ii;
            end
        end
        if(detchan==0)
            for ii=1:max(size(mda.scan.detectors))
                fprintf([num2str(mda.scan.detectors(ii).number+1) ': ' ...
                    mda.scan.detectors(ii).name '\n']);
            end
            error('Detector NOT FOUND: check above list');
        end
    end
    
    %% Create data matrix for 2D scan
    if((scan_dim==3 && xrfscan)||(scan_dim==2 && xrfscan==0))
        dataout = zeros(size(mda.scan.sub_scans,2),size(mda.scan.sub_scans(1).detectors_data,1),3);
        ccdnums = zeros(size(mda.scan.sub_scans,2),size(mda.scan.sub_scans(1).detectors_data,1));
        %display(imagechan);
        for ii=1:size(mda.scan.sub_scans,2)
            dataout(ii,:,1)=mda.scan.sub_scans(ii).detectors_data(:,detchan);
            dataout(ii,:,2)=mda.scan.positioners_data(ii,1);
            dataout(ii,:,3)=mda.scan.sub_scans(ii).positioners_data(:,1);
            if(imagechan>0)
                ccdnums(ii,:) = mda.scan.sub_scans(ii).detectors_data(:,imagechan);
            end
        end
        
        if(plotflag)
            % Create 2D figure
            if(isempty(mda.scan.positioners(1).description))
                ylab=mda.scan.positioners(1).name;
            else
                ylab=mda.scan.positioners(1).description;
            end
            if(isempty(mda.scan.sub_scans(1).positioners(1).description))
                xlab=mda.scan.sub_scans(1).positioners(1).name;
            else
                xlab=mda.scan.sub_scans(1).positioners(1).description;
            end
            if(isempty(mda.scan.sub_scans(1).detectors(detchan).description))
                detname=mda.scan.sub_scans(1).detectors(detchan).name;
            else
                detname=mda.scan.sub_scans(1).detectors(detchan).description;
            end
            clf;
            %hfig = surface(dataout(1,:,3),dataout(:,1,2),dataout(:,:,1)); colormap hot; shading interp;
            hfig = imagesc(dataout(1,:,3),dataout(:,1,2),dataout(:,:,1)); colormap hot;
            colormap hot; colorbar;
            set(gcf, 'PaperPosition', [.25 6.75 4 4.5]);
            set(gcf, 'Color' ,'w');
            axis fill tight;
            set(gca, 'YDir', 'normal');
            ylabel([ylab '  (' mda.scan.positioners.unit ')'], 'Interpreter', 'none', 'FontSize', 12);
            xlabel([xlab '  (' mda.scan.sub_scans(1).positioners.unit ')'], 'Interpreter', 'none', 'FontSize', 12);
            title(['File: ' filename '  Detector: ' detname], 'Interpreter', 'none', 'FontSize', 10);
            
            % Add CCD image numbers and pass structure to data cursor
            pass2click.mda = mda;
            pass2click.data = dataout(:,:,1);
            %ccdnums = zeros(size(dataout(:,:,1)));
            %startnum = 552924;
            %imsperpt = 20;
            %for ii=1:size(ccdnums,1)
            %    for jj=1:size(ccdnums,2)
            %        ccdnums(ii,jj) = startnum+imsperpt-1+(jj-1)*imsperpt+(ii-1)*size(ccdnums,2)*imsperpt;
            %    end
            %end
            pass2click.ccdnums = ccdnums;
            pass2click.spectra = xrfscan;
            pass2click.xaxis = [min(dataout(1,:,3)) max(dataout(1,:,3)) size(dataout(1,:,3),2)];
            pass2click.yaxis = [min(dataout(:,1,2)) max(dataout(:,1,2)) size(dataout(:,1,2),1)];
            set(hfig, 'UserData', pass2click);
            datacursormode on;
            dcm_obj = datacursormode(gcf);
            set(dcm_obj, 'DisplayStyle', 'window');
            set(dcm_obj, 'UpdateFcn', @click4ccd);
            %set(dcm_obj, 'UpdateFcn', @click4var);
        end
            %Analyze CCD centroids
            if(analyzeflag)
                Xcent=zeros(size(ccdnums));
                Ycent=zeros(size(ccdnums));
                SumInt=zeros(size(ccdnums));
                %imstart = 121241;
                h=waitbar(0,'Calculating centroids');
                for ii=1:size(ccdnums,1)
                    waitbar(ii/size(ccdnums,1));
                    for jj=1:size(ccdnums,2)
                        mdanum = mda.scan_number;%check this +1 for 1D scans
                        %filename = ['scan_' num2str(mdanum) '_img_' num2str(ccdnums(ii,jj),'%6.6d') '.tif'];
%                         filename=['../Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_' num2str(ccdnums(ii,jj), '%6.6d') '.tif'];
                        filename=['./Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(ccdnums(ii,jj)-1, '%6.6d') '.tif'];
                        
                        ccd = double(imread(filename));
                        ccd = ccd.*(ccd>0);
                        % hot pixel removal UNCOMMENT HERE
                        %{
                        ccd1 = zeros(size(ccd,1),size(ccd,2),4);
                        ccd1(:,:,1) = circshift(ccd,[0 1]);
                        ccd1(:,:,2) = circshift(ccd,[1 0]);
                        ccd1(:,:,3) = circshift(ccd,[0 -1]);
                        ccd1(:,:,4) = circshift(ccd,[-1 0]);
                        ccd2 = median(ccd1,3);
                        ccdmask = ccd>(ccd2+10);   %CHANGE THRESHOLD HERE
                        ccd = ccd.*(1-ccdmask)+ccd2.*ccdmask;
                        %}
                        if ii == 1
                            if jj == 1
                                showccd(filename,1,0);
                                if ~exist('ROIXstart','var')
                                    display('Please select rectangular ROI on image...');
                                    newROI = getrect;
                                    ROIXstart = round(newROI(1))
                                    ROIYstart = round(newROI(2))
                                    ROIXwidth = round(newROI(3))
                                    ROIYwidth = round(newROI(4))
                                end
                                rectangle('Position',[ROIXstart,ROIYstart,ROIXwidth,ROIYwidth],'edgecolor','y');
                            end
                        end
                        
                        ROI1=ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth));
                        SumInt(ii,jj)=sum(sum(ROI1));
                        line1=sum(ROI1,1);  % horizontal
                        line2=sum(ROI1,2);  % vertical
                        for kk=1:size(line1,2)
                            Xcent(ii,jj)=Xcent(ii,jj)+kk*line1(kk)/SumInt(ii,jj);
                        end
                        Xcent(ii,jj)=Xcent(ii,jj)+ROIXstart;
                        for kk=1:size(line2,1)
                            Ycent(ii,jj)=Ycent(ii,jj)+kk*line2(kk)/SumInt(ii,jj);
                        end
                        Ycent(ii,jj)=Ycent(ii,jj)+ROIYstart;
                        ROIRecord(ii,jj,:,:) = ROI1;
                    end

                end
                close(h);
                figure(501);clf;imagesc(dataout(1,:,3),dataout(:,1,2),SumInt);axis square tight;colormap hot; shading interp; title('Intensity');set(gca, 'YDir', 'normal');colorbar;
                figure(502);clf;imagesc(dataout(1,:,3),dataout(:,1,2),Xcent);axis square tight;colormap hot; shading interp; title('ROI X Centroid');set(gca, 'YDir', 'normal');colorbar;
                figure(503);clf;imagesc(dataout(1,:,3),dataout(:,1,2),Ycent);axis square tight;colormap hot; shading interp; title('ROI Y Centroid');set(gca, 'YDir', 'normal');colorbar;
                %figure(204);clf;waterfall(dataout(1,:,3),dataout(:,1,2),Ycent);axis fill tight;colormap default; shading interp; title('ROI Y Centroid');
            end
    end
    
    %% Create data matrix for 1D scan
    if((scan_dim==2 && xrfscan)||(scan_dim==1 && xrfscan==0))
        dataout = zeros(size(mda.scan.detectors_data,1),2);
        ccdnums = zeros(size(mda.scan.detectors_data,1),1);
        dataout(:,1) = mda.scan.positioners_data(:,1);
        dataout(:,2) = mda.scan.detectors_data(:,detchan);
        
        if(imagechan>0)
            ccdnums = mda.scan.detectors_data(:,imagechan);
        end
        
        if(plotflag)
            % Create 1D figure
            if(isempty(mda.scan.positioners(1).description))
                xlab=mda.scan.positioners(1).name;
            else
                xlab=mda.scan.positioners(1).description;
            end
            if(isempty(mda.scan.detectors(detchan).description))
                detname=mda.scan.detectors(detchan).name;
            else
                detname=mda.scan.detectors(detchan).description;
            end
            clf;
            if(xlab(1:6)=='Spiral')
                %display(xposchan);display(yposchan);
                hfig = scatter(mda.scan.detectors_data(:,xposchan),mda.scan.detectors_data(:,yposchan),...
                    300,dataout(:,2),'filled');
                set(gcf, 'PaperPosition', [.25 6.75 4 4.5]);
                set(gcf, 'Color' ,'w');
                axis square tight;
                xlabel('Hybrid X (um)', 'Interpreter', 'none', 'FontSize', 12);
                ylabel('Hybrid Y (um)', 'Interpreter', 'none', 'FontSize', 12);
                title(['File: ' filename '  Detector: ' detname], 'Interpreter', 'none', 'FontSize', 10);
                colorbar;
                % Add CCD image numbers and pass structure to data cursor
                pass2click.mda = mda;
                pass2click.data = dataout(:,2);
                pass2click.ccdnums = ccdnums;
                pass2click.spectra = xrfscan;
                pass2click.xaxis = mda.scan.detectors_data(:,xposchan);
                pass2click.yaxis = mda.scan.detectors_data(:,yposchan);
                set(hfig, 'UserData', pass2click);
                datacursormode on;
                dcm_obj = datacursormode(gcf);
                set(dcm_obj, 'DisplayStyle', 'window');
                set(dcm_obj, 'UpdateFcn', @click4ccd_spiral);
            else    
               % hfig = plot(dataout(:,1),dataout(:,2),'-o','LineWidth',2,'MarkerSize',8);
                hfig = plot(dataout(:,1),dataout(:,2),'LineWidth',2,'MarkerSize',8);
                set(gcf, 'PaperPosition', [.25 6.75 4 4.5]);
                set(gcf, 'Color' ,'w');
                axis square tight;
                xlabel([xlab '  (' mda.scan.positioners.unit ')'], 'Interpreter', 'none', 'FontSize', 12);
                title(['File: ' filename '  Detector: ' detname], 'Interpreter', 'none', 'FontSize', 10);
                
                % Add CCD image numbers and pass structure to data cursor
                pass2click.mda = mda;
                pass2click.data = dataout(:,2);
                pass2click.ccdnums = ccdnums;
                pass2click.spectra = xrfscan;
                pass2click.xaxis = [min(dataout(:,1)) max(dataout(:,1)) size(dataout,1)];
                set(hfig, 'UserData', pass2click);
                datacursormode on;
                dcm_obj = datacursormode(gcf);
                set(dcm_obj, 'DisplayStyle', 'window');
                set(dcm_obj, 'UpdateFcn', @click4ccd_line);
            end
            %Analyze CCD centroids
            if(analyzeflag)

                % Check for definition of ROI and create if not existant
                mdanum = mda.scan_number;%check this +1 for 1D scans
                filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(ccdnums(1), '%6.6d') '.tif'];
                ccd = double(imread(filename));
                % On the first pass, check to see if the ROI has been defined.  If not prompt for user to
                % select one and then continue
                if ~exist('ccd2','var')
                    ccd2=zeros(size(ccd));
                end
                showccd(filename,0,0);
                if ~exist('ROIXstart','var')
                    display('Please select rectangular ROI on image ...');
                    newROI = getrect;
                    ROIXstart = round(newROI(1))
                    ROIYstart = round(newROI(2))
                    ROIXwidth = round(newROI(3))
                    ROIYwidth = round(newROI(4))
                end
                rectangle('Position',[ROIXstart,ROIYstart,ROIXwidth,ROIYwidth],'edgecolor','y');
                
                Xcent=zeros(size(ccdnums));
                Ycent=zeros(size(ccdnums));
                Xfig = zeros(ROIXwidth+1,size(ccdnums,1));
                Yfig = zeros(ROIYwidth+1,size(ccdnums,1));
                SumInt=zeros(size(ccdnums));
                h=waitbar(0,'Calculating centroids');
                for ii=1:size(ccdnums,1)
                    waitbar(ii/size(ccdnums,1));
                    mdanum = mda.scan_number;%check this +1 for 1D scans
                        filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(ccdnums(ii)-1, '%6.6d') '.tif'];
                        ccd = double(imread(filename));ccd = ccd.*(ccd>0);
                        % hot pixel removal UNCOMMENT HERE
                        %%{
                        ccd1 = zeros(size(ccd,1),size(ccd,2),4);
                        ccd1(:,:,1) = circshift(ccd,[0 1]);
                        ccd1(:,:,2) = circshift(ccd,[1 0]);
                        ccd1(:,:,3) = circshift(ccd,[0 -1]);
                        ccd1(:,:,4) = circshift(ccd,[-1 0]);
                        ccd2 = median(ccd1,3);
                        ccdmask = ccd>(ccd2+5);   %CHANGE THRESHOLD HERE
                        ccd = ccd.*(1-ccdmask)+ccd2.*ccdmask;
                        %}
                    
                    % Create the ROI
                    ROI1=ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth));
                    SumInt(ii)=sum(sum(ROI1));
                    line1=sum(ROI1,1); % horizontal
                    Xfig(:,ii)=line1(1,:);
                    line2=sum(ROI1,2);  % vertical
                    Yfig(:,ii)=line2(:);
                    for kk=1:size(line1,2)
                        Xcent(ii)=Xcent(ii)+kk*line1(kk)/SumInt(ii);
                    end
                    Xcent(ii)=Xcent(ii)+ROIXstart;
                    for kk=1:size(line2,1)
                        Ycent(ii)=Ycent(ii)+kk*line2(kk)/SumInt(ii);
                    end
                    Ycent(ii)=Ycent(ii)+ROIYstart;
                    ROIRecord(ii,:,:) = ROI1;
                    
                end
                close(h);
                %figure(304);clf;plot(dataout(:,1),SumInt(:),'-o','LineWidth',2,'MarkerSize',8);title('Intensity');
                figure(304);clf;plot(dataout(:,1),SumInt(:),'LineWidth',2,'MarkerSize',8);title('Intensity');
                figure(305);clf;plot(dataout(:,1),Xcent(:),'-o','LineWidth',2,'MarkerSize',8);title('ROI X Centroid');
                figure(306);clf;plot(dataout(:,1),Ycent(:),'-o','LineWidth',2,'MarkerSize',8);title('ROI Y Centroid');
                figure(307);clf;imagesc(Xfig);colormap hot; axis tight square off;title('X CCD trace versus scan');
                figure(308);clf;imagesc(Yfig);colormap hot;axis tight square off;title('Y CCD trace versus scan');
                figure(304);clf;plot(dataout(:,1),SumInt(:),'-o','LineWidth',2,'MarkerSize',8);title('Intensity');
           end
        end
    end
    
    
    
end

%% Expanded output using cells
% instead of passing a matrix as output pass a cell containing more
% information than is possible.  This allows for only one variable per scan
% in the Matlab workspace and enables scan2scan analysis.  Only the
% components of conventional dataout matrix needs to be handled seperately
% for 1D or 2D data, the others can be passed as matrices or various size.
if (expandedOutputflag)
    outputCell.ROI = ROIRecord;
    % Parse the output cell for 1D scan
    if((scan_dim==2 && xrfscan)||(scan_dim==1 && xrfscan==0))
        outputCell.rowVar = dataout(:,1);
        outputCell.detReading = dataout(:,2);
    end
    
    % Parse the output cell for 2D scan
    if((scan_dim==3 && xrfscan)||(scan_dim==2 && xrfscan==0))
        outputCell.rowVar = dataout(:,:,3);
        outputCell.colVar =dataout(:,:,2);
        outputCell.detReading = dataout(:,:,1);
    end
    
    outputCell.xCent = Xcent;
    outputCell.yCent = Ycent;        
    outputCell.sumInt = SumInt;
    
    clear dataout
    dataout = outputCell;
end
