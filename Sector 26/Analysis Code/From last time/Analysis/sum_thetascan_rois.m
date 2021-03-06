function ROI_intensity=sum_thetascan_rois(scannums,plotflag)

if(nargin<2) 
    plotflag = 0;
end
%% Edit detector channels here
XRFchan = 39;   %set to Cs
imagechan = 21; %Pilatus image index channel
thetachan = 52;
twothetachan = 51;
gammachan = 50;
rdetchan = 49;
normchan = 1;

%% Edit ROI here
%ROIXstart = 158; ROIXwidth = 200; ROIYstart = 158; ROIYwidth = 200; %111
%central box, large, NW1
%ROIXstart = 255; ROIXwidth = 150; ROIYstart = 320; ROIYwidth = 190; 
%bottom box
% ROIXstart = 272; ROIXwidth = 240; ROIYstart = 20; ROIYwidth = 480; % This
% is ROI for the LSMO set 2 peak. 
%ROIXstart = 150; ROIXwidth = 250; ROIYstart = 5; ROIYwidth = 500; % Nanowire guess 08/18

% CsPbBr Ortho 310 ROI Pilatus
% ROIXstart = 60;
% ROIXwidth = 100;
% ROIYstart = 1;
% ROIYwidth = 75;

ROIXstart = 1;
ROIXwidth = 486;
ROIYstart = 1;
ROIYwidth = 194;

ROIs = [456, 17, 117, 15; 458, 15, 95, 114-95; 352, 13, 27, 13; 460, 14, 76, 13; 461, 12, 78, 8];

        
%% Code start
data_out.thvals = zeros(max(size(scannums)),1);

XRF1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],XRFchan,0,0);
norm1 = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],normchan,0,0);
twotheta = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],twothetachan,0,0);
gamma = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],gammachan,0,0);
rdet = loadmda(['mda/26idbSOFT_' num2str(scannums(1),'%4.4d') '.mda'],rdetchan,0,0);
for ii =1:size(XRF1,1)
    for jj=1:size(XRF1,2)
        data_out.ii(ii).jj(jj).im = zeros(ROIYwidth+1,ROIXwidth+1);
        data_out.ii(ii).jj(jj).immaxrc = zeros(ROIYwidth+1,ROIXwidth+1);
    end
end
chi2 = zeros(10,10);
I0 = sum(sum(norm1(:,:,1)))/(size(norm1,1)*size(norm1,2)); %add normalization
XRF1(:,:,1) = I0*XRF1(:,:,1)./norm1(:,:,1);
data_out.curve=zeros(2,max(size(scannums)));
data_out.Intensity = zeros(size(XRF1,1),size(XRF1,2));
data_out.XCent = zeros(size(XRF1,1),size(XRF1,2));
data_out.YCent = zeros(size(XRF1,1),size(XRF1,2));
data_out.thmax = zeros(size(XRF1,1),size(XRF1,2));
data_out.thcen = zeros(size(XRF1,1),size(XRF1,2));
data_out.ROIXstart = ROIXstart;
data_out.ROIYstart = ROIYstart;
data_out.ROIXwidth = ROIXwidth;
data_out.ROIYwidth = ROIYwidth;
data_out.twotheta = twotheta(1,1,1);
data_out.gamma = gamma(1,1,1);
data_out.rdet = rdet(1,1,1);


%%

%%
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
    XRF2(:,:,1) = I0*XRF2(:,:,1)./norm2(:,:,1);
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
    if(mm>1)
    trim1 = round(size(chi2,1)/2);
    trim2 = round(size(chi2,2)/2);
    range1 = trim1:(size(XRF1,1)-trim1);
    range2 = trim2:(size(XRF1,2)-trim2);
     for jjj = 1:size(chi2,1)
        for kkk = 1:size(chi2,2)
            %chi2(jj,kk) = sum(sum((XRF1(:,:,1) - circshift(XRF2(:,:,1),[jj-10,kk-10])).^2));
            chi2(jjj,kkk) = sum(sum((XRF1(range1,range2,1) - circshift(XRF2(range1,range2,1),[jjj-round(size(chi2,1)/2),kkk-round(size(chi2,2)/2)])).^2));
        end
    end
    [xcen1,ycen1,intemp] = find(chi2==min(min(chi2)));
    %figure;imagesc(chi2)
    %display(xcen1-round(size(chi2,1)/2));display(ycen1-round(size(chi2,2)/2))
    %display([xcen,ycen,xcen-round(size(chi2,1)/2),ycen-round(size(chi2,2)/2)])
    data_out.scan(mm).XRF = circshift(XRF2,[xcen1-round(size(chi2,1)/2),ycen1-round(size(chi2,2)/2)]);
    data_out.scan(mm).ccdnums = circshift(ccdnums,[xcen1-round(size(chi2,1)/2),ycen1-round(size(chi2,2)/2)]);
    else
        data_out.scan(mm).XRF=XRF2;
        data_out.scan(mm).ccdnums = ccdnums;
    end
    %}
       %{
    % line up with manually adjusted difference minimization
        %define a window to compare scans in coordinates of first scan
    range1 = 23:67;
    range2 = 39:62;
        %click on a representative point in all scans  INPUT HERE
   %feature_location = [57 49;59 44;59 39;61 29;62 19;62 15; 54 9; 55 14; 55 14];%scans 44-49
   %feature_location = [51 19;51 15;51 11;52 14;51 19;50 18;51 15;50 15;50 13;50 9;51 19;50 17];%scans 213-222 224 227
%    feature_location = [14 12; 14 13; 14 12; 14 12; 14 13; 14 12]; %scan of A feature scans 62-80
%   feature_location = [26 5; 31 6 ; 34 6 ; 37 5 ; 39 7 ; 43 6 ; 45 7 ; 46
%   8 ]; % scan 278 - 311 

    % scans 620 - 626: using the top part of Cr circle
    %feature_location = [25 1;26 1; 25 1;25 1;25 1]; 
    
    %scans 635-651 -- automatic lineup is very good, so this works:
    feature_location = [1 1; 1 1; 1 1; 1 1; 1 1; 1 1;];
    
    %scans 768,771    % feature_location = [1 1; 1 1; 1 1; 1 1; 1 1; 1 1;];
%     feature_location = [1 1 ; 1 1 ; 1 1 ; 1 1 ; 1 1 ; 1 1];

% feature location for scan starting with 1057 (signal level ~19)
   % feature_location = [ 18 5 ; 10 2 ; 1 1 ; 20 3 ; 17 3 ; 3 2  ];
   
   % feature location for maps: [1319 1322 1325  1328 1331]
   %feature_location = [1 1;1 1; 1 1;1 1; 1 1];
   
%    feature_location= [1 1; 1 1; 1 1; 1 1; 1 1; 1 1];

% Feature location for thscan 1428-1443
%    feature_location = [ 17 4 ; 16 4 ; 14 4 ; 15 4 ; 17 4 ; 16 4 ];

% Feature location for thscan [1677 1680 1683 1686 1700 1702]:
   % feature_location = [10 14;10 11; 12 13 ; 17 18; 19 14;19 15];
   
     st1 = -feature_location(mm,1)+feature_location(1,1);
    st2 = -feature_location(mm,2)+feature_location(1,2);
    %{
    st1 = -feature_location(mm,1)+feature_location(1,1)-round(size(chi2,1)/2);
    st2 = -feature_location(mm,2)+feature_location(1,2)-round(size(chi2,2)/2);
     for jj = 1:size(chi2,1)
        for kk = 1:size(chi2,2)
            %chi2(jj,kk) = sum(sum((XRF1(:,:,1) - circshift(XRF2(:,:,1),[jj-10,kk-10])).^2));
            chi2(jj,kk) = sum(sum((XRF1(range1,range2,1) - circshift(XRF2(range1,range2,1),[jj+st1,kk+st2])).^2));
        end
    end
    [xcen,ycen,intemp] = find(chi2==min(min(chi2)));
    %xcen=10;ycen=10;
    data_out.scan(mm).XRF = circshift(XRF2,[xcen+st1,ycen+st2]);
    data_out.scan(mm).ccdnums = circshift(ccdnums,[xcen+st1,ycen+st2]);
    %}
    data_out.scan(mm).XRF = circshift(XRF2,[st1,st2]);
    data_out.scan(mm).ccdnums = circshift(ccdnums,[st1,st2]);
    %}
    %%{
    ROI_intensity = zeros(size(XRF1,1), size(XRF1,2), size(ROIs,1));
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
            dummyArray = zeros( ROIYwidth+1, ROIXwidth+1 );

%             [ ROIXwidth ROIYwidth ]
            try
                ccd = double( imread(filename) );
            catch
                ccd = dummyArray;
            end
            
            ccd1=I0*ccd./norm2(ii,jj,1);
            ccd=ccd.*(ccd1>0);
            
            for kk = 1:size(ROIs, 1);

        %             size( ccd )
                    % End modification by Sid


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
                    ROI_intensity(ii,jj,kk) = sum(sum(ccd(ROIs(kk,3):ROIs(kk,3)+ROIs(kk,4), ROIs(kk,1):ROIs(kk,1)+ROIs(kk,2))));    %intensity of ccd in ROI region
                    
                    %identify and save image from maximum of rocking curve at each
                    %position
                    %data_out.curve(2,mm) = data_out.curve(2,mm) +sum(sum(ccd(ROIYstart:(ROIYstart+ROIYwidth),ROIXstart:(ROIXstart+ROIXwidth))));
            
        end
    end
    %}
end
end
close(h);
end



