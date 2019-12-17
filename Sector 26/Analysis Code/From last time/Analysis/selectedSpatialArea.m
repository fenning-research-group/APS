function selectedArea = selectedSpatialArea(thetaScanOutput,xSt,xEd,ySt,yEd)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    selectedArea = thetaScanOutput;
    selectedArea.Intensity = thetaScanOutput.Intensity(xSt:xEd,ySt:yEd);
    selectedArea.XCent = thetaScanOutput.XCent(xSt:xEd,ySt:yEd);
    selectedArea.YCent = thetaScanOutput.YCent(xSt:xEd,ySt:yEd);
    selectedArea.thmax = thetaScanOutput.thmax(xSt:xEd,ySt:yEd);
    selectedArea.thcen = thetaScanOutput.thcen(xSt:xEd,ySt:yEd);
    
    numSc = selectedArea.scan;
    for i=1:length(numSc)
        selectedArea.scan(i).XRF = thetaScanOutput.scan(i).XRF(xSt:xEd,ySt:yEd,:);
        selectedArea.scan(i).ccdnums = thetaScanOutput.scan(i).ccdnums(xSt:xEd,ySt:yEd,:);
    end
    
end

