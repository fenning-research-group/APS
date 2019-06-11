classdef apslog < handle
    properties
        filepath
    end
    
    properties (GetAccess = private)
        sectionidx = 0
        subsectionidx = 0
        subsubsectionidx = 0
    end
    
    methods
        function obj = setpath(obj, filepath)
        % Set filepath for log file to be written to
            if nargin < 2
                folder = uigetdir();    
                fname = input('Filename (no extension): ', 's');
                fname = [fname '.csv'];
                
                filepath = fullfile(folder, fname);
            end
            
            obj.filepath = filepath;
        end
        
        function obj = section(obj, scannum, title, flag)
        % Start new section

            if nargin < 4
                flag = 'none';
            end
            
            fid = fopen(obj.filepath, 'a');
            
            obj.sectionidx = obj.sectionidx + 1;
            obj.subsectionidx = 1;
            obj.subsubsectionidx = 1;
            
            writestring = sprintf(  '%d,%d,%d,%d,%s,%s\n',...
                                                        obj.sectionidx,...
                                                        obj.subsectionidx,...
                                                        obj.subsubsectionidx,...
                                                        scannum,...
                                                        title,...
                                                        flag...
                                  );
            
            fprintf(fid, writestring);
            fclose(fid);
        end
        
        function obj = subsection(obj, scannum, title, flag)
        % Start new subsection

            if nargin < 4
                flag = 'none';
            end
            
            fid = fopen(obj.filepath, 'a');
            
            obj.subsectionidx = obj.subsectionidx + 1;
            obj.subsubsectionidx = 1;
            
            writestring = sprintf(  '%d,%d,%d,%d,%s,%s\n',...
                                                        obj.sectionidx,...
                                                        obj.subsectionidx,...
                                                        obj.subsubsectionidx,...
                                                        scannum,...
                                                        title,...
                                                        flag...
                                  );
            
            fprintf(fid, writestring);
            fclose(fid);
        end
        
        function obj = subsubsection(obj, scannum, title, flag)
        % Start new subsubsection

            if nargin < 4
                flag = 'none';
            end
            
            fid = fopen(obj.filepath, 'a');
            
            obj.subsubsectionidx = obj.subsubsectionidx + 1;
            
            writestring = sprintf(  '%d,%d,%d,%d,%s,%s\n',...
                                                        obj.sectionidx,...
                                                        obj.subsectionidx,...
                                                        obj.subsubsectionidx,...
                                                        scannum,...
                                                        title,...
                                                        flag...
                                  );
            
            fprintf(fid, writestring);
            fclose(fid);
        end
        
        function obj = scan(obj, scannum, title, flag)
        % Add entry for a specific scan

            if nargin < 4
                flag = 'none';
            end
            
            fid = fopen(obj.filepath, 'a');
                       
            writestring = sprintf(  '%d,%d,%d,%d,%s,%s\n',...
                                                        obj.sectionidx,...
                                                        obj.subsectionidx,...
                                                        obj.subsubsectionidx,...
                                                        scannum,...
                                                        title,...
                                                        flag...
                                  );
            
            fprintf(fid, writestring);
            fclose(fid);
        end
        
        function tableofcontents(obj)
            ending_scan_idx = 500;
            
            fid = fopen(obj.filepath, 'r');
            raw = textscan(fid, '%d %d %d %d %s %s', 'delimiter', ',');
            
            sections = raw{1};
            subsections = raw{2};
            subsubsections = raw{3};
            scans = raw{4};
            titles = raw{5};
            flags = raw{6};
                        
            num_entries = length(sections);
            
            
            %% get the start/stop scan indices for each section

            section = 0;
            subsection = 0;
            subsectioncounter = 0;
            subsubsection = 0;
            subsubsectioncounter = 0;
                
            startscan = 1;
            sectionranges = [];
            subsectionranges = [];
            subsubsectionranges = [];
            
            
            for idx = 1:num_entries
                if sections(idx) > section
                    subsection = 1;
                    subsubsection = 1;
                    section = sections(idx);
                    
                    sectionranges(section, 1) = scans(idx);
                    if section > 1
                        sectionranges(section-1, 2) = scans(idx) - 1;
                    end     
                    
                elseif subsections(idx) > subsection
                    subsection = subsections(idx);
                    subsubsection = 1;
                    
                    subsectioncounter = subsectioncounter + 1;
                    
                    subsectionranges(subsectioncounter, 1) = scans(idx);
                    if subsectioncounter > 1
                        subsectionranges(subsection-1, 2) = scans(idx) - 1;
                    end
                    
                elseif subsubsections(idx) > subsubsection
                    subsubsection = subsubsections(idx);                    
                                        
                    subsubsectioncounter = subsubsectioncounter + 1;
                    
                    subsubsectionranges(subsubsectioncounter, 1) = scans(idx);
                    if subsectioncounter > 1
                        subsubsectionranges(subsubsection-1, 2) = scans(idx) - 1;
                    end                    
                end
            end            
            
            sectionranges(end,2 ) = ending_scan_idx;
            subsectionranges(end, 2) = ending_scan_idx;
            subsubsectionranges(end, 2) = ending_scan_idx;
            
            %% output the table of contents to the console
            
            section = 0;
            subsection = 0;
            subsectioncounter = 0;
            subsubsection = 0;
            subsubsectioncounter = 0;
                
            startscan = 1;           
            
            for idx = 1:num_entries
                % parse flag (same regardless of which level it applies to)
                if strcmp(flags{idx}, 'none')
                    flagstr = '';
                else
                    flagstr = sprintf('\tflag: %s', flags{idx});
                end
                
                % sort which level we are looking at
                if sections(idx) > section
                    subsection = 1;
                    subsubsection = 1;
                    section = sections(idx);
                  
                    fprintf('\n\n%s (%d:%d):\t%s', titles{idx}, sectionranges(section, 1), sectionranges(section,2), flagstr);
                    
                elseif subsections(idx) > subsection
                    subsection = subsections(idx);
                    subsubsection = 1;
                    
                    subsectioncounter = subsectioncounter + 1;
                    
                    fprintf('\n\t%s (%d:%d):\t%s', titles{idx}, subsectionranges(subsectioncounter, 1), subsectionranges(subsectioncounter,2), flagstr);

                elseif subsubsections(idx) > subsubsection
                    subsubsection = subsubsections(idx);
                    
                    subsubsectioncounter = subsubsectioncounter + 1;
                    
                    fprintf('\n\t\t%s (%d:%d):\t%s', titles{idx}, subsectionranges(subsubsectioncounter, 1), subsubsectionranges(subsubsectioncounter,2), flagstr);
                    
                else
                    fprintf('\n\t\t\t%s (%d):\t%s', titles{idx}, flagstr);
                end
            end
            
            fprintf('\n\n');
        end
    end
end
