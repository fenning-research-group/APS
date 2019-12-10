function f = findangles(x, phiflag, anglein, bragg_polar, sam_polar)

%Function for finding angles - generates q vector from sample angles, then
%compares its angle to the polar angle (returns an error if not the same)
%syntax should be something like:
%>> x=fminsearch(@(x) findangles(x,0,-90*pi/180,40*pi/180,5*pi/180), [0]);

if(phiflag)
    samth=anglein;
    samphi=x;
else
    samth=x;
    samphi=anglein;
end

vec1 = [cos(sam_polar) sin(sam_polar) 0];
vec2 = [vec1(1) vec1(2)*cos(samphi) vec1(2)*sin(samphi)];
vec3 = [(vec2(1)*cos(samth)+vec2(3)*sin(samth)) vec2(2) (-vec2(1)*sin(samth)+vec2(3)*cos(samth))];
%display(vec3);

f = (acos(vec3*[0 0 -1]')-bragg_polar)^2;

end