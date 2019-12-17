% mdaload.m  -  MATLAB routine for loading MDA files
% version 0.1  -  Dohn Arms (dohnarms@anl.gov)

function x = mdaload(y)
[fileID,errmsg] = fopen( y,'r','b');
if fileID < 0
    error( errmsg)
end
x.version = float32grab(fileID);
x.scan_number = int32grab(fileID);
x.data_rank = int16grab(fileID);
x.dimensions = fread(fileID,x.data_rank,'int32=>int32');
x.regular = int16grab(fileID);
extra_offset = int32grab(fileID);
x.scan = scangrab( fileID);
fseek( fileID, extra_offset, 'bof');
x.extra = extragrab(fileID);
fclose(fileID);
end

function scan = scangrab(fileID)
scan.scan_rank = int16grab(fileID);
scan.requested_points = int32grab(fileID);
scan.last_point = int32grab(fileID);
if scan.scan_rank > 1
    offsets = fread(fileID,scan.requested_points,'int32=>int32');
end
scan.name = strgrab(fileID);
scan.time = strgrab(fileID);
scan.number_positioners = int16grab(fileID);
scan.number_detectors = int16grab(fileID);
scan.number_triggers = int16grab(fileID);
if scan.number_positioners > 0
    for i = 1:scan.number_positioners
        scan.positioners(i) = posgrab(fileID);
    end
else
    scan.positioners = [];
end
if scan.number_detectors > 0
    for i = 1:scan.number_detectors
        scan.detectors(i) = detgrab(fileID);
    end
else
    scan.detectors = [];
end
if scan.number_triggers > 0
    for i = 1:scan.number_triggers
        scan.triggers(i) = triggrab(fileID);
    end
else
   scan.triggers = []; 
end
scan.positioners_data = fread(fileID, [scan.requested_points,scan.number_positioners],'float64=>float64');
scan.detectors_data = fread(fileID, [scan.requested_points,scan.number_detectors],'float32=>float32');
if scan.scan_rank > 1
    for i = 1:scan.last_point
        fseek( fileID, offsets(i), 'bof');
        scan.sub_scans(i) = scangrab(fileID);
    end
else
    scan.subscans = [];
end
end


function pos = posgrab(fileID)
pos.number = int16grab(fileID);
pos.name = strgrab(fileID);
pos.description = strgrab(fileID);
pos.step_mode = strgrab(fileID);
pos.unit = strgrab(fileID);
pos.readback_name = strgrab(fileID);
pos.readback_description = strgrab(fileID);
pos.readback_unit = strgrab(fileID);
end

function det = detgrab(fileID)
det.number = int16grab(fileID);
det.name = strgrab(fileID);
det.description = strgrab(fileID);
det.unit = strgrab(fileID);
end

function trig = triggrab(fileID)
trig.number = int16grab(fileID);
trig.name = strgrab(fileID);
trig.command = float32grab(fileID);
end

function extra = extragrab(fileID)
extra.number_pvs = int16grab(fileID);
for i = 1:extra.number_pvs
    extra.pvs(i) = pvgrab(fileID);
end
end

function pv = pvgrab(fileID)
pv.name = strgrab(fileID);
pv.descr = strgrab(fileID);
type = int16grab(fileID);
switch type
    case 0
        pv.type = 'char';
    case 29
        pv.type = 'int16';
    case 30
        pv.type = 'float32';
    case 32
        pv.type = 'int8';
    case 33
        pv.type = 'int32';
    case 34
        pv.type = 'float64';
end
if type > 0
    pv.count = int16grab(fileID);
    pv.unit = strgrab(fileID);
else
    pv.count = 1;
    pv.unit = '';
end
switch type
    case 0
        pv.values = strgrab(fileID);
    case 29
        pv.values = fread(fileID,pv.count,'int16=>int16');
        if mod(pv.count,2) == 1
            fread(fileID,1,'int16=>int16');
        end
    case 30
        pv.values = fread(fileID,pv.count,'float32=>float32');
    case 32
        pv.values = fread(fileID,pv.count,'int8=>int8');
        len = mod(pv.count,4);
        if len > 0
            fread(fileID,4-len,'int8=>int8');
        end
    case 33
        pv.values = fread(fileID,pv.count,'int32=>int32');
    case 34
        pv.values = fread(fileID,pv.count,'float64=>float64');
end
end

function val = int32grab(fileID)
val = fread(fileID,1,'int32=>int32');
end

function val = int16grab(fileID)
val = fread(fileID,1,'int32=>int16');
end

function val = float32grab(fileID)
val = fread(fileID,1,'float32=>float32');
end

function str = strgrab(fileID)
len1=fread(fileID,1,'int32=>int16');
if len1 > 0
    len2=fread(fileID,1,'int32=>int16');
    str=fread(fileID,[1,len2],'char=>char');
    m = mod(len2,4);
    if m > 0
        fread(fileID,4-m,'char=>char');
    end
else
    str='';
end
end


