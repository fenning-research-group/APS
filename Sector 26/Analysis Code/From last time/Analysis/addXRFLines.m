function addXRFLines_rek(elms)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%% with textscan
%{
lookuptable_fpath = '/CNMshare/savedata/2018R3/20181023/Analysis/xrf_emission_table.txt';

edges = cell(length(elms),1);


fid = fopen(lookuptable_fpath, 'r');
readline = fgetl(fid);
readline = textscan(readline, '%s\t');
edgelabels = readline{1}(3:end);

readline = fgetl(fid);
readline = fgetl(fid);

while readline ~= -1
    raw_read = textscan(readline, '%s\t');
    line_components = raw_read{1};
    for elm_idx = 1:numel(elms)
        if strcmp(elms{elm_idx}, line_components(2))
            edges{elm_idx} = line_components(3:end);
        end
    end
    readline = fgetl(fid);
end

fclose(fid);
hold on;
colors = lines(length(elms));

for elm_idx = 1:length(elms)
%     disp(elms{elm_idx});
    max_keV = max(xlim);
    plotcount = 0;
    for edge_idx = 1:length(edges{elm_idx})
        edge_keV = str2num(edges{elm_idx}{edge_idx});
%         disp(edge_keV);
        if edge_keV <= max_keV            
            plot_height = max(ylim) - (max(ylim)-min(ylim))*(0.2*elm_idx-0.05*plotcount);
            text_height = max(ylim) - (max(ylim)-min(ylim))*(0.2*elm_idx - 0.01 - 0.05*plotcount);
            plot([edge_keV edge_keV], [min(xlim) plot_height], ':','color', colors(elm_idx,:));
            text(edge_keV + 0.1, text_height, sprintf('%s %s', elms{elm_idx}, edgelabels{edge_idx}), 'color', colors(elm_idx,:));
            plotcount = plotcount + 1;
        end
    end
end

%}
    
%% with struct
load('/CNMshare/savedata/2018R3/20181023/Analysis/xrfLines.mat');

edgelabels = xrfLines.lineNames;


hold on;
max_keV = max(xlim);

colors = lines(length(elms));

for elm_idx = 1:length(elms)
    plotcount = 0;
    edges = xrfLines.(elms{elm_idx}).xrfEmissionLines;
    for edge_idx = 1:length(edges)
%         disp(edge_keV);
        if edges(edge_idx) <= max_keV            
            plot_height = max(ylim) - (max(ylim)-min(ylim))*(0.01 + 0.01*(elm_idx-1)+0.06*plotcount+0.02);
            text_height = max(ylim) - (max(ylim)-min(ylim))*(0.001 + 0.01*(elm_idx-1)+0.06*plotcount+0.02);
            plot([edges(edge_idx) edges(edge_idx)], [min(xlim) plot_height], ':','color', colors(elm_idx,:), 'linewidth', 2);
            peak_label = edgelabels(edge_idx, :);
            if sum(peak_label == 'a')
                peak_label = strrep(peak_label, 'a', '\alpha_');
            elseif sum(peak_label == 'b')
                peak_label = strrep(peak_label, 'b', '\beta_');
            elseif sum(peak_label == 'g')
                peak_label = strrep(peak_label, 'g', '\gamma_');
            end
            text(edges(edge_idx) + 0.1, text_height, sprintf('%s %s', elms{elm_idx}, peak_label), 'color', colors(elm_idx,:), 'fontsize', 14);
            plotcount = plotcount + 1;
        end
    end
end

end

