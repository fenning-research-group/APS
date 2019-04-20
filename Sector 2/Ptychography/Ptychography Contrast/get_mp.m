function response = get_mp(varargin)
    preamble = 'https://www.materialsproject.org/rest/v2';
    api_key = 'fyqhoW8JXH7GT6wb';       %api key for rek010@eng.ucsd.edu
    
    totalquery = preamble;
    
    for idx = 1:numel(varargin)
        totalquery = strcat(totalquery, '/', varargin{idx});
    end
    
    totalquery = strcat(totalquery, '?API_KEY=', api_key);
    
    response = webread(totalquery);
end