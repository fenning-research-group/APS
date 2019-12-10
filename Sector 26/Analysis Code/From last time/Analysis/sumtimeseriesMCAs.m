function mcaSpectrum = timeseriesMCAs(fileName)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
mdaData = mdaload(fileName);

%mca0, mca1, mca2, mca3
calSlope =[1.0537000e-02,1.0540000e-02,1.0561e-2, 1.0535000e-02]; %1.1994972e-02 mca8 val
calOffset = [-6.9206998e-02,-7.2107002e-02,-8.085100e-02, -7.0338003e-02]; %,4.2292913e-03 mca8 val
mcaNames={'mca0','mca1','mca2','mca3'};
mcaPVnames={'26idcXMAP:mca1.VAL','26idcXMAP:mca2.VAL','26idcXMAP:mca3.VAL',...
    '26idcXMAP:mca4.VAL'};
mca0 = zeros([2048 1]);
mca1 = zeros([2048 1]);
mca2 = zeros([2048 1]);
mca3 = zeros([2048 1]);

for mcaIndex=1:length(mcaNames)
    mcaEvStr = genvarname([mcaNames{mcaIndex} 'eV']);
    mcaEv = [0:1:2047].*calSlope(mcaIndex)+calOffset(mcaIndex);
    eval([mcaEvStr '=mcaEv;']);
end

rquPts = mdaData.scan.requested_points;
% mcaSumSum = zeros([2048 1]);
mcaData = zeros(2048, numel(mcaNames));

for i = 1:rquPts
    subScan = mdaData.scan.sub_scans(i);
    for g=1:length(mcaPVnames)
        detectors = subScan.detectors;
        for gg=1:length(detectors)
            det = subScan.detectors(gg).name;
            if strcmp(det,mcaPVnames{g})
%                 mcaData{i}(:,g) = subScan.detectors_data(:,gg);
                mcaData(:,g) = mcaData(:,g) + subScan.detectors_data(:,gg);
            end
        end
    end    
end


%% interpolation code
% mca_ev = the eV after scaling 0:2047 by the calibration numbers. 
% mca_readings = the ccd readings at 0:2047. ev_interp is the new eV range 
% to interpolate ccd readings to

ev_interp = [0:5:10e3]/1e3;             %dummy ev range from 0:10 keV
mcaData_interp = zeros(numel(ev_interp), numel(detectors));

mcaData_interp(:,1) = interp1(mca0eV, mcaData(:,1), ev_interp); 
mcaData_interp(:,2) = interp1(mca1eV, mcaData(:,2), ev_interp); 
mcaData_interp(:,3) = interp1(mca2eV, mcaData(:,3), ev_interp); 
mcaData_interp(:,4) = interp1(mca3eV, mcaData(:,4), ev_interp); 

mcaSum = sum(mcaData_interp, 2);
mcaSpectrum = [ev_interp', mcaSum];
%

f = figure ('Name',[fileName]);
plot(ev_interp, mcaSum);
title({sprintf('Filename: %s', fileName), 'Time Series Integrated 4-Element Detector Counts'})
xlabel('Energy (keV)')
ylabel('Detector Counts');

end


