function txt=click4ccd(empt, event_obj)

pos = get(event_obj, 'Position');
himage=get(event_obj, 'Target');
info = get(himage, 'UserData');

jj = round((pos(1)-info.xaxis(1))*(info.xaxis(3)-1)/(info.xaxis(2)-info.xaxis(1))+1);
ii = round((pos(2)-info.yaxis(1))*(info.yaxis(3)-1)/(info.yaxis(2)-info.yaxis(1))+1);

ccdnum=info.ccdnums(ii,jj);
data = info.data(ii,jj);

if(ccdnum>0)
    mdanum = info.mda.scan_number;
    %for kk = 1:20
        %ccdnum=ccdnum-(kk-1);
    filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_Pilatus_' num2str(ccdnum, '%6.6d') '.tif'];

  %display(filename)
    showccd(filename,1);
    %{
    tempmat = zeros(236,116,20);
    figure(2000);
    for kk = 1:20
        ccdnum=ccdnum-(kk-1);
        filename=['Images/' num2str(mdanum) '/scan_' num2str(mdanum) '_img_' num2str(ccdnum, '%6.6d') '.tif'];
         ccd = double(imread(filename));
             ccd1 = zeros(size(ccd,1),size(ccd,2),4);
             ccd1(:,:,1) = circshift(ccd,[0 1]);
             ccd1(:,:,2) = circshift(ccd,[1 0]);
             ccd1(:,:,3) = circshift(ccd,[0 -1]);
             ccd1(:,:,4) = circshift(ccd,[-1 0]);
             ccd2 = median(ccd1,3);
             ccdmask = ccd>(ccd2+100);   %CHANGE THRESHOLD HERE
             ccd = ccd.*(1-ccdmask)+ccd2.*ccdmask;
%              display(size(tempmat));
%              display(size(ccd(135:370,195:310)))
             tempmat(:,:,kk) = ccd(135:370,195:310);
    end
%     for kk=1:20
%         imagesc(tempmat(:,:,kk));
%         axis image
%         pause(0.1)
%     end
    %pause(0.1)
    %end
    %}
end
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
%    plot(info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1000,1));
    plot(log(1+info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1200,1)));
    ylabel('CHANNEL ONE', 'Interpreter', 'none', 'FontSize', 12);
    subplot(4,1,2);
    %plot(info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1000,2));
    plot(log(1+info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1200,2)));
    ylabel('CHANNEL TWO', 'Interpreter', 'none', 'FontSize', 12);    
    subplot(4,1,3);
    %plot(info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1000,3));
    plot(log(1+info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1200,3)));
    ylabel('CHANNEL THREE', 'Interpreter', 'none', 'FontSize', 12);    
    subplot(4,1,4);
    %plot(info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1000,4));
    plot(log(1+info.mda.scan.sub_scans(ii).sub_scans(jj).detectors_data(1:1200,4)));
    ylabel('CHANNEL FOUR', 'Interpreter', 'none', 'FontSize', 12);
end
%}
txt = {
    ['X: ' num2str(pos(1)) '   Y:' num2str(pos(2)) '   Z:' num2str(data)], ...
    ['jj: ' num2str(jj) '   ii:' num2str(ii)] ...
    ['CCD: ' num2str(ccdnum)] ...
    };
