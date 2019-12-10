function mca_sum = PlotSingleMCAs(fileName)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
mcaNames={'mca0','mca1', 'mca2', 'mca3'};
calSlope =[1.0537000e-02,1.0540000e-02,1.0561e-2, 1.0535000e-02]; %1.1994972e-02 mca8 val
calOffset = [-6.9206998e-02,-7.2107002e-02,-8.085100e-02, -7.0338003e-02]; %,4.2292913e-03 mca8 val
mcaPVnames={'26idcXMAP:mca1.VAL','26idcXMAP:mca2.VAL','26idcXMAP:mca8.VAL',...
    '26idcXMAP:mca4.VAL'};

mcareadings = zeros(2048, numel(mcaNames));
ev_interp = [0:5:10e3]/1e3;

mca_ev = repmat([0:2047]', 1, 4);

figure, hold on;
title({'Individual mca Spectra', sprintf('%s', fileName)}, 'interpreter', 'none');
xlabel('Energy (eV)');
ylabel('Counts');

for i = 1:numel(mcaNames)
    fid = fopen([fileName '_' mcaNames{i}], 'r');
    eof = 0;
    while ~eof
        headerline = fgetl(fid);
        if strcmp(headerline, 'DATA: ')
            eof = 1;
        end
    end
    mca_ev(:,i) = mca_ev(:,i)*calSlope(i) + calOffset(i);
    readout = textscan(fid, '%d');
    mca_readings(:,i) = double(readout{1});
    mca_readings_interp(:,i) = interp1(mca_ev(:,i), mca_readings(:,i), ev_interp);
    plot(ev_interp, mca_readings_interp(:,i));
end

legend(mcaNames);
mca_sum = [ev_interp', sum(mca_readings_interp, 2)];

figure;
plot(ev_interp, mca_sum(:,2));
title({'Summed mca Spectra', sprintf('%s', fileName)}, 'interpreter', 'none');
xlabel('Energy (eV)');
ylabel('Counts');
%mca0, mca1, mca2, mca3


end

