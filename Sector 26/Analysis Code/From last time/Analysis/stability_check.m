% roi 3 filter 4 scans
scannum=[436,437,438];
firstimage=[210166;211563;212959];

% % roi 4 filter 3 scans
% scannum=[439,440];
% firstimage=[214355, 215751];
scan=struct;
for m=1:length(scannum)
    
temp=zeros(195,487);

scan.(['sum' num2str(scannum(m))])=zeros(195,487);

    for n=1:1000
    temp= double(imread(['./Images/' num2str(scannum(m)) '/scan_' num2str(scannum(m)) '_img_Pilatus_' num2str(firstimage(m)+n) '.tif']));
    temp(temp<5)=0;
    scan.(['sum' num2str(scannum(m))])=temp+scan.(['sum' num2str(scannum(m))]);
    end
    
%imwrite(scan.(['sum' num2str(scannum(m))]),['./Images/scan_' num2str(scannum(m)) '_sum.tif']);
figure('Name',num2str(scannum(m))),
imshow(scan.(['sum' num2str(scannum(m))]));hold on;
coreff(m)=corr2(scan.(['sum' num2str(scannum(1))]),scan.(['sum' num2str(scannum(m))]));
text(10,10,['corrletaion=' num2str(coreff(m))],'color','r');
end
