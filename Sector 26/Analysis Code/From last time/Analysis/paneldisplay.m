function figh = paneldisplay(scannums,detchan,fignum)
if(nargin<3)
    figh = figure;clf reset;colormap hot;colorbar;axis auto tight;hold on
else
    figh = figure(fignum);clf reset;colormap hot;colorbar;axis auto tight;hold on
end

for ii=1:max(size(scannums))
    datatemp = loadmda(['mda/26idbSOFT_' num2str(scannums(ii),'%4.4d') '.mda'],detchan,0,0);
    imagesc(datatemp(1,:,3)',datatemp(:,1,2),datatemp(:,:,1));axis auto;pause(0.1);
end
axis tight;
end
