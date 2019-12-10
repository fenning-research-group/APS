function txt=click4rock(empt, event_obj)

pos = get(event_obj, 'Position');
himage=get(event_obj, 'Target');
info = get(himage, 'UserData');
jj = round((pos(1)-info.xaxis(1))*(info.xaxis(3)-1)/(info.xaxis(2)-info.xaxis(1))+1);
ii = round((pos(2)-info.yaxis(1))*(info.yaxis(3)-1)/(info.yaxis(2)-info.yaxis(1))+1);
display(pos);display([ii jj])
figure(1001);
set(gcf, 'Name', 'Rocking curve');
pos=get(gcf,'Position');
set(gcf, 'Position', [pos(1:2) 800 1300]);
axes('Position', [0 0 1 1]);
set(gcf, 'PaperPosition', [.25 6.75 3 3]);
set(gcf, 'Color','w');
%display(info.data.thvals(1));display(info.data.ii(ii).jj(jj).rc);
subplot(2,1,2);
[thaxis ind] = sort(info.data.thvals(:));
plot(thaxis,info.data.ii(ii).jj(jj).rc(ind),'-','color','blue','LineWidth',2);
axis square tight;
title(['Rocking curve at x value:  ' num2str(pos(1)) ' y value: ' num2str(pos(2))]);
ccd = info.data.ii(ii).jj(jj).im;
%                        ccd1 = zeros(size(ccd,1),size(ccd,2),4);
%                         ccd1(:,:,1) = circshift(ccd,[0 1]);
%                         ccd1(:,:,2) = circshift(ccd,[1 0]);
%                         ccd1(:,:,3) = circshift(ccd,[0 -1]);
%                         ccd1(:,:,4) = circshift(ccd,[-1 0]);
%                         ccd2 = median(ccd1,3);
%                         ccdmask = ccd>(ccd2+10);   %CHANGE THRESHOLD HERE
%                         ccd = ccd.*(1-ccdmask)+ccd2.*ccdmask;
                        
subplot(2,1,1);
imagesc(ccd);axis image tight off;colormap hot;


%pause(.5);
%subplot(3,1,1);


txt = {
    ['X: ' num2str(pos(1)) '   Y:' num2str(pos(2))], ...
    ['jj: ' num2str(jj) '   ii:' num2str(ii)] ...
    };