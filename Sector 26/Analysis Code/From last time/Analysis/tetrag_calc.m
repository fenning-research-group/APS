%% General tetragonal/cubic/hexagonal beamline diffraction calculator 

% Given an orientation and lattice, what is the sample and detector
% positioning necessary to reach a desired reflection
% Surface normal will be +x outboard at th 0

Ekev = 9; % beam energy in kev
%Ekev = 8.04778; % beam energy in kev (lab source)

% FOR HEXAGONAL CASE THE FOLLOWING MUST ALL BE 4-VECTORS [h k i l]

%refl = [1 0 3];   %Desired reflection [h k l] CHANGE HERE
% refl = [1 0 -1 4];   %Desired reflection [h k i l] CHANGE HERE
refl = [0 0 0 1];   %Desired reflection [h k i l] CHANGE HERE
%refl = [0 1 -1 0];   %Desired reflection [h k i l] CHANGE HERE - 100 deg
%refl = [1 0 -1 0];   %Desired reflection [h k i l] CHANGE HERE - 100 deg

%lat_surf = [0 0 1];  % Surface normal vector (pointing outboard at th 0) in [h k i l] 
%lat_vert = [1 0 0];  % Vertical in-plane vector (pointing up at phi 0) in [h k i l]

lat_surf = [0 0 0 1];  % Surface normal vector (pointing outboard at th 0) in [h k i l] 
lat_vert = [1 0 -1 0]; % Vertical in-plane vector (pointing up at phi 0) in [h k i l]

lat = [3.073 3.073 10.053]; %4H-SiC
% lat = [3.073 3.073 15.11]; %6H-SiC
%lat = [3.905 3.905 3.905]; %LSMO
%lat = [5.326 5.352 7.5399]; %CaFeO3

%lat = [4.22 4.22 6.91]; %InGaAs nanowire
%lat = [5.6353 5.6353 6.93]; %GaAs nanowire

%lat = [3.905 3.905 3.85]; %LSMO
%lat = [3.567 3.567 3.567]; %Diamond
%lat = [5.835 5.835 5.835]; %Si
%lat = [8.38 8.38 8.38]; % Fe3O4 cubic lattice vectors a b c (A)
%lat = [3.36 3.36 5.9]; % TaS2 trigonal lattice vectors a b c (A)
%lat = [3.905 3.905 3.905]; % SrTiO3 cubic lattice vectors a b c (A)
%lat = [3.373 3.373 5.903]; % Sr2IrO4 tetragonal lattice vectors a b c (A)
%lat = [4.9424 4.9424 14.0161]; % V2O3 lattice vectors a b c (A)

lambda = 12.39842/Ekev; % wavelength 1/A
kbeam = 2*pi*Ekev/12.39842; % beam momentum 1/A
ki = kbeam*[0 0 1];        %Incident k vector (+z lab frame)


%tetragonal case
%{
astar = [2*pi/lat(1) 0 0];
bstar = [0 2*pi/lat(2) 0];
cstar = [0 0 2*pi/lat(3)];

q1 = refl(1)*astar+refl(2)*bstar+refl(3)*cstar;
surf_q = lat_surf(1)*astar+lat_surf(2)*bstar+lat_surf(3)*cstar;
vert_q = lat_vert(1)*astar+lat_vert(2)*bstar+lat_vert(3)*cstar;
%}

%rhombohedral case
%{
avec = [lat(1) 0 0];
bvec = [lat(2)*cosd(120) lat(2)*sind(120) 0];
cvec = [0 0 lat(3)];

vol1 = dot(avec,cross(bvec,cvec));
astar = 2*pi*cross(bvec,cvec)/vol1;
bstar = 2*pi*cross(cvec,avec)/vol1;
cstar = 2*pi*cross(avec,bvec)/vol1;

q1 = refl(1)*astar+refl(2)*bstar+refl(3)*cstar;
surf_q = lat_surf(1)*astar+lat_surf(2)*bstar+lat_surf(3)*cstar;
vert_q = lat_vert(1)*astar+lat_vert(2)*bstar+lat_vert(3)*cstar;
%}

%hexagonal case
%%{
avec = [lat(1) 0 0];
bvec = [lat(2)*cosd(120) lat(2)*sind(120) 0];
cvec = [0 0 lat(3)];

vol1 = dot(avec,cross(bvec,cvec));
%vol2 = sqrt(3)*lat(1)^2*lat(3)/2;
astar = 2*pi*cross(bvec,cvec)/vol1;
bstar = 2*pi*cross(cvec,avec)/vol1;
cstar = 2*pi*cross(avec,bvec)/vol1;

a1star = 2/3*astar-1/3*bstar;
a2star = -1/3*astar+2/3*bstar;
a3star = -1/3*astar-1/3*bstar;

q1 = refl(1)*a1star+refl(2)*a2star+refl(3)*a3star+refl(4)*cstar;
surf_q = lat_surf(1)*a1star+lat_surf(2)*a2star+lat_surf(3)*a3star+lat_surf(4)*cstar;
vert_q = lat_vert(1)*a1star+lat_vert(2)*a2star+lat_vert(3)*a3star+lat_vert(4)*cstar;
%}

surf_q_unit = surf_q./norm(surf_q);
inplane_q1 = q1-(q1*surf_q_unit')*surf_q_unit;
phi_offset = acos((inplane_q1*vert_q')/(norm(vert_q)*norm(inplane_q1)));

direction_check = surf_q_unit*(cross(inplane_q1,vert_q))';
if direction_check>0
    phi_offset = - phi_offset;
end

th_b = asin(lambda*norm(q1)/(4*pi));  %Bragg theta
th_p = pi/2 - th_b;         %Polar angle of bragg Q relative to beam (-z)

%Polar angle of reflection relative to surface normal vector
sam_polar = acos((q1*surf_q')/(norm(q1)*norm(surf_q)));

%%{
th_out=zeros(361,6);
clear ok_reg;  %Keeps track where reflection is accessible under theta
in_it_ok=0;num_reg_ok=0;
clear goodreg; %Keeps track where reflection makes it out the window
in_it_good=0;num_reg_good=0;

if(norm(inplane_q1)==0) %No phi dependence of reflection
    x=fminsearch(@(x) findangles(x,0,0,th_p,0), 0);
    th_out(:,1)=x*180/pi;
    th_out(:,2)=(1:361)-181;
    th_out(:,3)=x*180/pi;
    th_out(:,4)=2*x*180/pi;
    th_out(:,5)=0;
    num_reg_good=1;
    goodreg(1,1)=1;goodreg(1,2)=361;
    num_reg_ok=1;
    ok_reg(1,1)=1;ok_reg(1,2)=361;
else
for ii=-180:180
    %phi = 0 corresponds to total non-surface normal component oriented up (+y)
    %positive phi then rotates this following right-hand-rule about outboard (+x)
    %phi numbers correspond to a CCW rotation of sample relative to post when
    %viewed on bench microscope
    
    samphi = ii*pi/180-phi_offset;
    if samphi > pi
        samphi = samphi - 2*pi;
    end
    if samphi < -pi
        samphi = samphi + 2*pi;
    end
    th_out(ii+181,2) = ii;
    if((pi/2-acos(abs(sin(sam_polar)*cos(samphi))))<th_p)
        if(in_it_ok==0)
            in_it_ok=1;
            num_reg_ok=num_reg_ok+1;
            ok_reg(num_reg_ok,1)=ii+181;
        end
        if(ii==180)
            ok_reg(num_reg_ok,2)=ii+181;
        end
        x=fminsearch(@(x) findangles(x,0,samphi,th_p,sam_polar), 0);
        samth = x;
        surf_norm=[cos(samth) 0 -sin(samth)];
        vec1 = [cos(sam_polar) sin(sam_polar) 0];
        vec2 = [vec1(1) vec1(2)*cos(samphi) vec1(2)*sin(samphi)];
        vec3 = [(vec2(1)*cos(samth)+vec2(3)*sin(samth)) vec2(2) (-vec2(1)*sin(samth)+vec2(3)*cos(samth))];
        kf = ki+norm(q1)*vec3;
        th_out(ii+181,1) = x*180/pi;
        th_out(ii+181,3) = 90-acos((kf*surf_norm')/norm(kf))*180/pi;
        th_out(ii+181,4) = atan(kf(1)/kf(3))*180/pi;
        th_out(ii+181,5) = 90-acos((kf*[0 1 0]')/norm(kf))*180/pi;
        th_out(ii+181,6) = (th_out(ii+181,1)>=0)*(th_out(ii+181,3)>=0)*(th_out(ii+181,4)>=0)*(th_out(ii+181,5)>=-2); %substrate-exit angle check
        %th_out(ii+181,6) = (th_out(ii+181,1)>=0)*(th_out(ii+181,4)>=0)*(th_out(ii+181,5)>=-2); %wire-no sample exit angle check
    else
        if(in_it_ok==1)
            in_it_ok=0;
            ok_reg(num_reg_ok,2)=ii+181-1;
        end
    end
    if((in_it_good == 0)&&(th_out(ii+181,6)==1))
        in_it_good=1;
        num_reg_good=num_reg_good+1;
        goodreg(num_reg_good,1)=ii+181;
    end
    if((in_it_good==1)&&((th_out(ii+181,6)==0)||(ii==180)))
        in_it_good=0;
        goodreg(num_reg_good,2)=ii+181-1;
    end
    
end
end

figure(61);clf;
set(gcf, 'Name', ['DIFFRACTOMETER POSITIONS FOR ' num2str(refl) ]);
pos=get(gcf,'Position');
set(gcf, 'Position', [pos(1:2) 1000 1000]);
axes('Position', [0 0 1 1]);
set(gcf, 'PaperPosition', [.25 6.75 3 3]);
set(gcf, 'Color','w');
subplot(2,2,1);hold on;
for ii=1:num_reg_ok
plot(th_out(ok_reg(ii,1):ok_reg(ii,2),2),th_out(ok_reg(ii,1):ok_reg(ii,2),1),'LineWidth',2,'color','blue');axis square tight
end
for ii=1:num_reg_good
plot(th_out(goodreg(ii,1):goodreg(ii,2),2),th_out(goodreg(ii,1):goodreg(ii,2),1),'LineWidth',2,'color','green');
end
ylabel('SAMPLE THETA (deg)', 'Interpreter', 'none', 'FontSize', 12);
title('SAMPLE THETA', 'Interpreter', 'none', 'FontSize', 14);
xlabel('SAMPLE PHI (deg)', 'Interpreter', 'none', 'FontSize', 12);
subplot(2,2,2);hold on;
for ii=1:num_reg_ok
plot(th_out(ok_reg(ii,1):ok_reg(ii,2),2),th_out(ok_reg(ii,1):ok_reg(ii,2),4),'LineWidth',2,'color','blue');axis square tight
end
for ii=1:num_reg_good
plot(th_out(goodreg(ii,1):goodreg(ii,2),2),th_out(goodreg(ii,1):goodreg(ii,2),4),'LineWidth',2,'color','green');
end
ylabel('DETECTOR TWO THETA (deg)', 'Interpreter', 'none', 'FontSize', 12);
xlabel('SAMPLE PHI (deg)', 'Interpreter', 'none', 'FontSize', 12);
title('DETECTOR TWO THETA', 'Interpreter', 'none', 'FontSize', 14);
subplot(2,2,3);hold on;
for ii=1:num_reg_ok
plot(th_out(ok_reg(ii,1):ok_reg(ii,2),2),th_out(ok_reg(ii,1):ok_reg(ii,2),5),'LineWidth',2,'color','blue');axis square tight
end
for ii=1:num_reg_good
plot(th_out(goodreg(ii,1):goodreg(ii,2),2),th_out(goodreg(ii,1):goodreg(ii,2),5),'LineWidth',2,'color','green');
end
ylabel('DETECTOR GAMMA (deg)', 'Interpreter', 'none', 'FontSize', 12);
xlabel('SAMPLE PHI (deg)', 'Interpreter', 'none', 'FontSize', 12);
title('DETECTOR GAMMA', 'Interpreter', 'none', 'FontSize', 14);
subplot(2,2,4);hold on;
for ii=1:num_reg_ok
plot(th_out(ok_reg(ii,1):ok_reg(ii,2),2),th_out(ok_reg(ii,1):ok_reg(ii,2),3),'LineWidth',2,'color','blue');axis square tight
end
for ii=1:num_reg_good
plot(th_out(goodreg(ii,1):goodreg(ii,2),2),th_out(goodreg(ii,1):goodreg(ii,2),3),'LineWidth',2,'color','green');
end
ylabel('SAMPLE EXIT ANGLE (deg)', 'Interpreter', 'none', 'FontSize', 12);
xlabel('SAMPLE PHI (deg)', 'Interpreter', 'none', 'FontSize', 12);
title('SAMPLE EXIT ANGLE', 'Interpreter', 'none', 'FontSize', 14);

%}


