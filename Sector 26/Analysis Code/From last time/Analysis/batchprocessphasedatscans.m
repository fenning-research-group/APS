%% File Saving Settings:

    rootpath = 'G:\My Drive\FRG\Projects\APS\26IDC3\20190530 Phase Mapping\correctvalues_twothetatolerance_0pt02';
    
    if ~exist(rootpath, 'dir')
        mkdir(rootpath);
    end
    
    %initialize output struct
    
    n_experiments = 10;
    output(1:n_experiments) = struct(   'thvals', [],...
                                        'summed_diffraction', [],...
                                        'xrfmaps', [],...
                                        'ii', [],...
                                        'curve', [],...
                                        'Intensity', [],...
                                        'XCent', [],...
                                        'YCent', [],...
                                        'thmax', [],...
                                        'thcen', [],...
                                        'ROIXstart', [],...
                                        'ROIYstart', [],...
                                        'ROIXwidth', [],...
                                        'ROIYwidth', [],...
                                        'twotheta', [],...
                                        'gamma', [],...
                                        'rdet', [],...
                                        'scan_overlap', [],...
                                        'scan', [],...
                                        'materials', [],...
                                        'bad_pixels', [],...
                                        'experiment_id', []...
                                    );
%% XRF detector settings: 
    %% Scan 1-168
    % Cs 39 R18
    % Pb 42 R17
    % Br 41 R26
    % Cr 38 R9
    % Ca 37 R6
    % I/Eu 40 R17

    xrfchans1 = [39, 42, 41, 38, 37, 40];
    xrflabels1 = {'Cs', 'Pb', 'Br', 'Cr', 'Ca', 'I_Eu'};
    %% Scans >= 188
    % Cs 11
    % Pb 12
    % Br 13
    % Cr 14
    % Ca 15
    % Eu 16
    scannum = 1;
    xrfchans2 = [11, 12, 13, 14, 15, 16];
    xrflabels2 = {'Cs', 'Pb', 'Br', 'Cr', 'Ca', 'Eu'};

%% initialize pilatus data
    if ~exist('qbin')
        qbin = qbin_pilatus('./Images/image_Pilatus_000001.tif');
        close all;
    end

    tic
%% Pristine CsPbBr3 Crystals
    materials = {'PbBr2',...
                 'CsPbBr3',...
                 'CsBr',...
                 'EuBr2',...
                };
            
    material_twotheta = {   [18.6000   20.9600   21.6500   22.0600   23.7200   23.9700   28.7900   29.0100   29.1100   30.2100   30.6100   33.9600   34.6900   34.7700   35.9800   37.7200   38.0300   38.4600 39.4000   39.7200   40.8700   42.6600   44.1200   44.1800   44.4000   44.9900   46.0400   48.5300   49.3200   51.2200   51.6500   52.9800   53.3500   53.8700   54.6400   55.2000  55.9200   57.5500   58.0000   58.0300]...
                            [10.0921   10.8589   10.9918   11.6666   12.6788   14.2943   15.0815   15.2349   18.8543   19.4166   21.5023   21.5432   22.4532   23.4245   24.2117   25.1830   25.4693   26.3997 26.4611   27.4426   27.6982   28.6490   29.7635   30.4076   30.4792   30.7245   30.7961   33.3624   34.1701   34.2621   34.3950   34.4768   35.4481   37.5543   37.8610   39.0470  43.8000   46.6000   49.0000   49.5000   54.3000   56.9000]...
                            [20.6700   29.4000   36.2100   42.0500   47.3000   52.1400],...
                            [12.4800   14.6500   21.2600   21.7300   24.3300   25.1100   26.2800   27.4300   29.5400   30.5600   30.5800   32.8500   34.3500   34.3700   34.6800   35.2400   36.9900   40.2400 41.0400   42.0100   44.0500   47.1400   49.8600   52.8200   56.2300]...
                        };
        
scantitles = {'CsPbBr3 Random Crystal', 'CsPbBr3 Target Crystal 1', 'CsPbBr3 Target Crystal 2', 'CsPbBr3 Target Crystal 3'};
scannumbers = {[18:2:26], [49:2:57], [85:2:97], [131:2:141]};
XRFchan = [39, 39, 39, 39]; %Cs
          
for ii = 1:1
    writepath = fullfile(rootpath, scantitles{ii});
    if ~exist(writepath, 'dir')
        mkdir(writepath);
    end
    
    tempoutput = phasemap_registered_interpolated(scannumbers{ii}, 0, qbin, XRFchan(ii), materials, material_twotheta, xrflabels1, xrfchans1);
    tempoutput.experiment_id = scantitles{ii};
%     save(fullfile(writepath, [scantitles{ii} '_phasemap.mat']), 'output');
    close all;
    
    plotphasedat(tempoutput, writepath);
    plot2thetas(qbin, tempoutput);
    export_fig(fullfile(writepath, ['_summed_Pilatus_ccd']), '-m3', '-painters');
    close all;
    output(scannum) = tempoutput;
    scannum = scannum + 1;
    
end
% 
% %% 5% Eu CsPbBr3 Crystals
% scannum = 5;
%     materials = {'PbBr2',...
%                  'CsPbBr3',...
%                  'CsBr',...
%                  'EuBr2',...
%                 };
%             
%     material_twotheta = {   [18.6000   20.9600   21.6500   22.0600   23.7200   23.9700   28.7900   29.0100   29.1100   30.2100   30.6100   33.9600   34.6900   34.7700   35.9800   37.7200   38.0300   38.4600 39.4000   39.7200   40.8700   42.6600   44.1200   44.1800   44.4000   44.9900   46.0400   48.5300   49.3200   51.2200   51.6500   52.9800   53.3500   53.8700   54.6400   55.2000  55.9200   57.5500   58.0000   58.0300]...
%                             [10.0921   10.8589   10.9918   11.6666   12.6788   14.2943   15.0815   15.2349   18.8543   19.4166   21.5023   21.5432   22.4532   23.4245   24.2117   25.1830   25.4693   26.3997 26.4611   27.4426   27.6982   28.6490   29.7635   30.4076   30.4792   30.7245   30.7961   33.3624   34.1701   34.2621   34.3950   34.4768   35.4481   37.5543   37.8610   39.0470  43.8000   46.6000   49.0000   49.5000   54.3000   56.9000]...
%                             [20.6700   29.4000   36.2100   42.0500   47.3000   52.1400],...
%                             [12.4800   14.6500   21.2600   21.7300   24.3300   25.1100   26.2800   27.4300   29.5400   30.5600   30.5800   32.8500   34.3500   34.3700   34.6800   35.2400   36.9900   40.2400 41.0400   42.0100   44.0500   47.1400   49.8600   52.8200   56.2300]...
%                         };
% 
% scantitles = {'CsPbEuBr3 Target Crystal 1', 'CsPbEuBr3 Target Crystal 2 ROI 1', 'CsPbEuBr3 Target Crystal 2 ROI 2', 'CsPbEuBr3 Target Crystal 3'};
% scannumbers = {[203:2:219, 223:2:229], [251:2:259, 263, 268], [251:2:259, 263, 268], [328:2:338, 342, 344]};
% XRFchan = [11, 11, 11, 11]; %Cs
%           
% for ii = 1:length(XRFchan)
%     writepath = fullfile(rootpath, scantitles{ii});
%     if ~exist(writepath, 'dir')
%         mkdir(writepath);
%     end
%     
%     tempoutput = phasemap_registered_interpolated(scannumbers{ii}, 0, qbin, XRFchan(ii), materials, material_twotheta, xrflabels2, xrfchans2);
%     tempoutput.experiment_id  = scantitles{ii};
% %     save(fullfile(writepath, [scantitles{ii} '_phasemap.mat']), output);
%     close all;
%     
%     plotphasedat(tempoutput, writepath);
%     plot2thetas(qbin, tempoutput);
%     export_fig(fullfile(writepath, ['_summed_Pilatus_ccd']), '-m3', '-painters');
%     close all;
%     
%     output(scannum) = tempoutput;
%     scannum = scannum + 1;
% end

%% 5% Eu MAPI Film
    scannum = 9;
    
    materials = {'PbI2',...
                 'EuI',...
                 'EuMAPbI3'...
                };
            
    material_twotheta = {   [23,12, 30.53, 35.17, 42.53, 37.07, 40.18, 34,41, 47.26, ]...
                            [25.96, 32.74, 25.87, 38.44, 31.86, 27.24, 18.79, 30.75, 42.25, 46.97, 42.48, 21.61]...
                            [14.1936, 20.1053, 28.4923, 38.0248]...
                        };


scantitles = {'Eu Doped MAPI ROI 1', 'Eu Doped MAPI ROI 2'};
scannumbers = {[448:2:456],[203:2:219, 223:2:229]};
XRFchan = [23, 23];       %Cr channel to register to the Cr/Pt pattern
          
for ii = 1:1
    writepath = fullfile(rootpath, scantitles{ii});
    if ~exist(writepath, 'dir')
        mkdir(writepath);
    end
    
    tempoutput = phasemap_registered_interpolated(scannumbers{ii}, 0, qbin, XRFchan(ii), materials, material_twotheta, xrflabels2, xrfchans2);
    tempoutput.experiment_id  = scantitles{ii};
%     save(fullfile(writepath, [scantitles{ii} '_phasemap.mat']), output);
    close all;
    
    plotphasedat(tempoutput, writepath);
    plot2thetas(qbin, tempoutput);
    export_fig(fullfile(writepath, ['_summed_Pilatus_ccd']), '-m3', '-painters');
    close all;
    
    output(scannum) = tempoutput;
    scannum = scannum + 1;
end

toc