function txt=click4ccd_line(empt, event_obj)

pos = get(event_obj, 'Position');
himage=get(event_obj, 'Target');
info = get(himage, 'UserData');

jj = round((pos(1)-info.xaxis(1))*(info.xaxis(3)-1)/(info.xaxis(2)-info.xaxis(1))+1);
%display(jj)
ccdnum=info.ccdnums(jj);
data = info.data(jj);
%display(ccdnum);display(data);display(jj);
if(ccdnum>0)
    mdanum = info.mda.scan_number;%check this +1 for 1D scans
    filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(ccdnum, '%6.6d') '.tif'];
    tempimg=showccd(filename,1);
end
%figure(22);clf;plot(log(1+sum(tempimg(185:300,150:310),1)),'LineWidth',2);

%{
% Display fluorescence spectra
if(info.spectra)
    figure(20000);clf;
    set(gcf, 'Name', 'MCA SPECTRA');
    pos=get(gcf,'Position');
    set(gcf, 'Position', [pos(1:2) 800 800]);
    axes('Position', [0 0 1 1]);
    set(gcf, 'PaperPosition', [.25 6.75 3 3]);
    set(gcf, 'Color','w');
    subplot(4,1,1);
    plot(info.mda.scan.sub_scans(jj).detectors_data(1:1000,1));
    ylabel('CHANNEL ONE', 'Interpreter', 'none', 'FontSize', 12);
    subplot(4,1,2);
    plot(info.mda.scan.sub_scans(jj).detectors_data(1:1000,2));
    ylabel('CHANNEL TWO', 'Interpreter', 'none', 'FontSize', 12);    
    subplot(4,1,3);
    plot(info.mda.scan.sub_scans(jj).detectors_data(1:1000,3));
    ylabel('CHANNEL THREE', 'Interpreter', 'none', 'FontSize', 12);    
    subplot(4,1,4);
    plot(info.mda.scan.sub_scans(jj).detectors_data(1:1000,4));
    ylabel('CHANNEL FOUR', 'Interpreter', 'none', 'FontSize', 12);
end
%}
txt = {
    ['X: ' num2str(pos(1)) '   Y:' num2str(data)], ...
    ['jj: ' num2str(jj)] ...
    ['CCD: ' num2str(ccdnum)] ...
    };
