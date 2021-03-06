function mcaSpectrum = showSelfDefineXRFROI(fileName,elms,deltaE,xSt,ySt,xEd,yEd,plotXRF)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
load('xrfLines.mat');
initDetNum = 26;
mdaData = mdaload(fileName);
tempData = loadmda(fileName,initDetNum,0,0,0,0);
tempMap = tempData(:,:,1);

% Spatial ROI range
if xSt==0 & xEd==0 & ySt==0 & yEd==0
    xSt=1;xEd=size(tempMap,2);
    ySt=1;yEd=size(tempMap,1);
elseif xEd > size(tempMap,2)
    ferror('xEd is largerer than the map.\n');
elseif yEd > size(tempMap,1)
    ferror('yEd is largerer than the map\n');
end


%mca0, mca1, mca2, mca3 energy cali
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

% find the elm line peak energy position
lineNames = xrfLines.lineNames;
for elmIndex=1:length(elms)
    elm = elms{elmIndex};
    dashIndex = findstr(elm,'_');
    lineType = elm(dashIndex(1)+1:end);
    if strcmp(lineType, 'L')
        xrfType = 'La1';
        lineIndex = 4;
    elseif strcmp(lineType,'K');
        xrfType = 'Ka1';
        lineIndex = 1;
    else
        xrfType = 'Ma1';
        lineIndex = 9;
    end
    
    elmXRFlines = xrfLines.(elm(1:dashIndex(1)-1)).xrfEmissionLines;
    roiLine = elmXRFlines(lineIndex);
    elmIntMap = NaN([yEd xEd]);
    
    elmlinesStr = genvarname([elm '_eV']);
    elmIntMapStr = genvarname([elm '_map']);
    
    eval([elmIntMapStr '=elmIntMap;']);
    eval([elmlinesStr '=roiLine;']);
end

% Look into each MCA and integrate the desired elemental emission line
% under the curve
ev_interp = [0:5:10e3]/1e3;             %dummy ev range from 0:10 keV
for elmIndex =1:length(elms)
    tempROImap = NaN([yEd xEd]);
    elm = elms{elmIndex};
    centerLine = eval([elm '_eV']);
    leftE = centerLine-deltaE;
    rightE = centerLine+deltaE;
    leftIndex = find(abs(ev_interp-leftE)<0.005);
    rightIndex = find(abs(ev_interp-rightE)<0.005);
    
    for i=ySt:yEd
        subScan = mdaData.scan.sub_scans(i).sub_scans;
        for j=xSt:xEd
            fourMCAperPix = zeros([size(ev_interp) 1]);
            for g=1:length(mcaPVnames)
                detectors = subScan(j).detectors;
                for gg=1:length(detectors)
                    det = detectors(gg).name;
                    if strcmp(det,mcaPVnames{g})
                        mcaData = subScan(j).detectors_data(:,gg);
                        mcaXval = eval([mcaNames{g} 'eV']);
                        mca_interp = interp1(mcaXval,mcaData, ev_interp);
                        fourMCAperPix = fourMCAperPix + mca_interp;
                        %                     eval([mcaNames{g} '= mcaData;']);
                    end
                end
            end
            
            % After getting the sum 4-mca per pix, now to extract the ROI
            % information
            tempROImap(i,j) = trapz(fourMCAperPix(leftIndex(1):rightIndex(1)));
        end
    end
    eval([elm '_map=tempROImap;']);
    mcaSpectrum.([elm '_map']) = tempROImap;
    if plotXRF==1
        f = figure('Name',elm);
        surf(tempROImap);shading flat; view(2); axis tight; colormap(hot);
    end
end


end


