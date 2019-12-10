function timeseriesMCAs(fileName)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
mdaData = mdaload(fileName);

numlines = 5;


%mca0, mca1, mca2, mca3
calSlope =[1.0537000e-02,1.0540000e-02,1.0561e-2, 1.0535000e-02]; %1.1994972e-02 mca8 val
calOffset = [-6.9206998e-02,-7.2107002e-02,-8.085100e-02, -7.0338003e-02]; %,4.2292913e-03 mca8 val
mcaNames={'mca0','mca1','mca2','mca3'};
mcaPVnames={'26idcXMAP:mca1.VAL','26idcXMAP:mca2.VAL','26idcXMAP:mca3.VAL',...
    '26idcXMAP:mca4.VAL'};
mcaEv = zeros([2048 4]);


for mcaIndex=1:length(mcaNames)
    mcaEv(:,mcaIndex) = [0:1:2047].*calSlope(mcaIndex)+calOffset(mcaIndex);
end

rquPts = mdaData.scan.requested_points;
% mcaSumSum = zeros([2048 1]);
mcaData = cell(rquPts, 1);
ev_interp = [0:5:10e3]/1e3;             %dummy ev range from 0:10 keV

for i = 1:rquPts
    mcaData{i} = zeros(numel(ev_interp), numel(mcaNames));
    subScan = mdaData.scan.sub_scans(i);
    for mcaIndex=1:length(mcaPVnames)
        detectors = subScan.detectors;
        for gg=1:length(detectors)
            det = subScan.detectors(gg).name;
            if strcmp(det,mcaPVnames{mcaIndex})
                mcaData{i}(:,mcaIndex) = interp1(mcaEv(:,mcaIndex), subScan.detectors_data(:,gg), ev_interp);
            end
        end
    end    
end


%% interpolation code
% mca_ev = the eV after scaling 0:2047 by the calibration numbers. 
% mca_readings = the ccd readings at 0:2047. ev_interp is the new eV range 
% to interpolate ccd readings to

f = figure ('Name',[fileName]);
hold on
title({sprintf('Filename: %s', fileName), 'Time Series Integrated 4-Element Detector Counts'})
xlabel('Energy (keV)')
ylabel('Detector Counts');
time_step = round(rquPts/numlines);

colors = parula(numlines);
plotoffset = max(mcaData{i}(:))*0.2;

plot(ev_interp, sum(mcaData{1},2) + numlines*plotoffset, 'color', colors(1,:));
hline = plot(xlim, [numlines*plotoffset numlines*plotoffset], ':', 'color', colors(1,:));
% hline.color(4) = 0.5;

plot(ev_interp, sum(mcaData{end},2), 'color', colors(end,:));

for i = 2:numlines-1
    plot(ev_interp, sum(mcaData{time_step*i},2) + (numlines - i)*plotoffset, 'color', colors(i,:)); 
    hline = plot(xlim, [numlines-i numlines-i]*plotoffset, ':', 'color', colors(1,:));
%     hline.color(4) = 0.5;
end

end