function ccd=showccd(filename,logflag,frac)

if(nargin<3); frac=1; end

ccd = double(imread(filename));
ccd = ccd.*(ccd>0);
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
%{
ccd4 = double(imread('Images/98/scan_98_img_046553.tif'));
ccd1 = zeros(size(ccd4,1),size(ccd4,2),4);
ccd1(:,:,1) = circshift(ccd4,[0 1]);
ccd1(:,:,2) = circshift(ccd4,[1 0]);
ccd1(:,:,3) = circshift(ccd4,[0 -1]);
ccd1(:,:,4) = circshift(ccd4,[-1 0]);
ccd2 = median(ccd1,3);
ccdmask = ccd4>(ccd2+2000);   %CHANGE THRESHOLD HERE
ccd5 = ccd4.*(1-ccdmask)+ccd2.*ccdmask;
%}
%ccd6 = double(imread('Images/72/scan_72_img_121753.tif'));
 %ccd1=ccd;
%ccd=ccd.*(ccd1<5000);
% NDarks = 5722:5732;
% Darks = 0;
% for n = 1:numel(NDarks)
%     Darks = Darks + double(imread(sprintf('Images/image_%06d.tif',NDarks(n))));
% end
% Darks = Darks / numel(NDarks);
%ccd = double(imread(filename))-double(imread('Images/image_005722.tif'));
%ccd = (ccd-ccd2).*((ccd-ccd2)>0);
% ccd = abs(double(imread(filename))-Darks);

%display(mean(mean(ccd(739:1350,56:107))));
%ccd=ccd*(1.4357e4)/mean(mean(ccd(725:1450,75:210)));
%ccd=ccd*(1.5722e4)/mean(mean(ccd(739:1350,56:107)));
%ccd=(ccd-ccd4).*(ccd3-ccd4)./(ccd2-ccd4);

%sz=size(ccd);

figure(10001);
clf
set(gcf, 'Name', 'CCD');
pos=get(gcf,'Position');

set(gcf, 'Position', [pos(1:2) 1600 800]);
axes('Position', [0 0 1 1]);
set(gcf, 'PaperPosition', [.25 6.75 3 3]);
set(gcf, 'Color','w');
axis off

if(nargin<2) logflag=0;end

%logflag=1; frac=.3;

if(~logflag)
    h=imagesc((ccd(:,:)));axis equal tight off;
    %caxis([0 10000]);
    %figure(12);clf; plot(log10(sum(ccd(:,:),1)));axis([100 400 2 6]);figure(10001);
    %caxis([min(min(ccd)) max(max(ccd))*frac]);
    %ca=caxis;
    %caxis([ca(1) ca(1)+frac*(abs(ca(2)-ca(1)))]);
    %caxis([600 800]);
    
    
else
  ccd(ccd<1)=1; 
  h=imagesc(log10(ccd)); axis equal tight off; caxis([0 1.5]);
  %figure(101);clf; plot(log10(sum(abs(ccd(120:370,200:300)),1)+1));
  %hold on;plot(log10(sum(ccd5(120:370,200:300),1)+1),'color','red');axis([1 100 1 6])
figure(10001);
    %caxis([min(min(log10(ccd))) max(max(log10(ccd)))*frac]);
    %ca=caxis;
    %caxis([ca(1) ca(1)+frac*(abs(ca(2)-ca(1)))]);
    %caxis([0 1000]);
end

text(50,50,[filename ',   max val: ' num2str(max(ccd(:)))],'Interpreter','none', 'color','w');
if (max(ccd(:)==65535)) text(50, 75, 'SATURATED', 'color','r'); end

set(h, 'UserData', ccd);
%datacursormode on;
dcm_obj = datacursormode(gcf);
set(dcm_obj, 'DisplayStyle', 'window');
set(dcm_obj, 'UpdateFcn', @click4value);

%----------------------------------------------
function txt=click4value(empt, event_obj)

pos = get(event_obj, 'Position');
%pixel=get(event_obj, 'DataIndex');
hfig=get(event_obj, 'Target');
%display(pos);
%display(pixel);
values = get(hfig, 'UserData');
val=values(pos(2),pos(1));

txt = { 
    ['pixel: ' num2str(pos(1)) '  ' num2str(pos(2))], ...
    [' val: ' num2str(val)] ...
    };