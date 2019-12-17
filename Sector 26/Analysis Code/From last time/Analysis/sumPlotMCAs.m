function [Eu_counts,mcaSpectrum] = sumPlotMCAs(fileName,detectorNum,xSt,ySt,xEd,yEd,plotXRF)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
mdaData = mdaload(fileName);
xrfData = loadmda(fileName,detectorNum,0,0,0,0);
xrfMap = xrfData(:,:,1);

if xSt == 0 && xEd == 0
    xSt = 1;
    xEd = size(xrfMap, 2);
end
if ySt == 0 && yEd == 0
    ySt = 1;
    yEd = size(xrfMap, 1);
end

if(xSt > xEd)
    a = xSt;
    xSt = XEd;
    XEd = a;
    clear a;
end

if(ySt > yEd)
    a = ySt;
    ySt = yEd;
    yEd = a;
    clear a;
end

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

if plotXRF == 1
    f = figure('Name',[fileName 'D-' int2str(detectorNum)]);
    title({sprintf('Detector#: %d, Filename: %s', detectorNum, fileName), 'Intensity Map of This Detector'})
    h_bg = surf(xrfMap);shading flat; view (2); axis equal; axis off; colormap gray;
    
    hold on
end

rquPts = mdaData.scan.requested_points;
% mcaSumSum = zeros([2048 1]);

for i=ySt:yEd
    subScan = mdaData.scan.sub_scans(i).sub_scans;
    for j=xSt:xEd
        for g=1:length(mcaPVnames)
            detectors = subScan(j).detectors;
            for gg=1:length(detectors)
                det = detectors(gg).name;
                if strcmp(det,mcaPVnames{g})
                    mcaData = subScan(j).detectors_data(:,gg);
                    eval([mcaNames{g} '=' mcaNames{g} '+mcaData;']);
                end
            end
        end
    end
end

%% interpolation code
% mca_ev = the eV after scaling 0:2047 by the calibration numbers.
% mca_readings = the ccd readings at 0:2047. ev_interp is the new eV range
% to interpolate ccd readings to


for i=ySt:yEd
    subScan = mdaData.scan.sub_scans(i).sub_scans;
    for j=xSt:xEd
        for g=1:length(mcaPVnames)
            detectors = subScan(j).detectors;
            for gg=1:length(detectors)
                det = detectors(gg).name;
                if strcmp(det,mcaPVnames{g})
                    mcaData = subScan(j).detectors_data(:,gg);
                    eval([mcaNames{g} '=mcaData;']);
                    ev_interp = [0:5:10e3]/1e3;             %dummy ev range from 0:10 keV
                    mca0_readings_interp = interp1(mca0eV, mca0, ev_interp);
                    mca1_readings_interp = interp1(mca1eV, mca1, ev_interp);
                    mca2_readings_interp = interp1(mca2eV, mca2, ev_interp);
                    mca3_readings_interp = interp1(mca3eV, mca3, ev_interp);
                    mcaSum = mca0_readings_interp + mca1_readings_interp + mca2_readings_interp + mca3_readings_interp;
                    mcaSpectrum = [ev_interp, mcaSum];
                    Eu_counts(i,j)=max(mcaSum(find(ev_interp>5.8 & ev_interp<5.9)));
                end
            end
        end
    end
end
figure,
imagesc(Eu_counts);


%sum spectrum
ev_interp = [0:5:10e3]/1e3;             %dummy ev range from 0:10 keV
mca0_readings_interp = interp1(mca0eV, mca0, ev_interp);
mca1_readings_interp = interp1(mca1eV, mca1, ev_interp);
mca2_readings_interp = interp1(mca2eV, mca2, ev_interp);
mca3_readings_interp = interp1(mca3eV, mca3, ev_interp);
mcaSum = mca0_readings_interp + mca1_readings_interp + mca2_readings_interp + mca3_readings_interp;
mcaSpectrum = [ev_interp, mcaSum];
Eu_counts=max(mcaSum(find(ev_interp>6.2 & ev_interp<6.4)))

f = figure ('Name',[fileName 'xSt' int2str(xSt) 'xEd' int2str(xEd)]);
plot(ev_interp, mcaSum);
title({sprintf('Detector#: %d, Filename: %s', detectorNum, fileName), 'Summed 4-Element Detector Counts'})
xlabel('Energy (keV)')
ylabel('Detector Counts');

end


