function findxrfpeak(eV, tol, plotflag)
    if nargin < 3
        plotflag = 0;
    end
    
    load('/CNMshare/savedata/2018R3/20181023/Analysis/xrfLines.mat');
    el_list = fieldnames(xrfLines);
    el_list(strcmp(el_list,'lineNames')) ='';
    hit_count = 0;
    
    edgelabels = xrfLines.lineNames;

    for elIndex =1:length(el_list)
        el_hit = 0;
        el = el_list{elIndex};
        emission_lines = xrfLines.(el).xrfEmissionLines;
        for line_idx = 1:numel(emission_lines)
            if abs(emission_lines(line_idx) - eV) <= tol
                fprintf('%s %s\t%.3f keV\n', el, edgelabels(line_idx,:), emission_lines(line_idx));
                if ~el_hit               
                    hit_count = hit_count + 1;
                    hit_list{hit_count} = el;
                    el_hit = 1;
                end
            end
        end
    end          
        
    if plotflag
        addXRFLines(hit_list);
    end
        
end