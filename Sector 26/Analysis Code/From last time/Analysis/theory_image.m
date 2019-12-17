function dataout = theory_image(Tthdet, Rdet, Gamdet, Samth, Samchi, Samphi, Samthick, Refl_d, Ekev, Intensity, plotflag)

%---------------------------------------------
%SCRIPT FOR CREATING THEORETICAL THIN FILM DIFFRACTION IMAGES
% All vector notation in beam convention [+outboard +up +downstream]
% Martin actually wrote this but please don't give him any credit at all


if(nargin<11) 
    plotflag=1; % output images
end

if(nargin<9) 
    Ekev = 10.0; % beam energy in kev
    Intensity = 1; %scaling for detector counts
end

if(nargin<4) 
    Samth =  18.786; % Sample rotation about vertical axis (deg, right-hand-rule)
    Samchi = .5; % Sample rotation about beam axis (deg, RHR)
    Samphi = 0; % Sample rotation about outboard axis (deg, RHR)
    Refl_d = 3.85*1; %lattice parameter in angstroms
    Samthick = 0.01; %Film thickness in microns
end

if(nargin<3) 
    Tthdet=37.46; % **Cylindrical** two theta angle to the center of the detector (deg)
    Rdet=800000;  % **Cylindrical** radius from sample to the center detector (um)
    Gamdet=0;   % **Cylindrical** out of diffraction plane angle to the center of the detector (deg)
end
pxsz = 55; % medipix pixel size (um)
Ndet = 516; % medipix number of pixels

Dzp = 150; % Zone plate diameter - um
drzp = 20; % Zone plate outermost zone - nm
Fzp = Dzp*drzp*Ekev/1.239842;  % Zone plate focal distance - um
Dstop = 65; % Zone plate central stop diameter - um

kb = 2*pi*Ekev/12.39842; % beam momentum 1/A
Num_emit = round((Samthick*1e4)/Refl_d); %number of mirror planes

scrsz = get(0,'ScreenSize');  % Plot size parameters
imd = scrsz(3)/5;
imb = scrsz(4)-imd;

Ncen = round(Ndet/2);
imaxis = (1-Ncen):(Ndet-Ncen);

%initialize output images
imbin = zeros(Ndet,Ndet);

%Detector center and pixel plane vectors
tthd = pi * Tthdet / 180;
gamd = pi * Gamdet / 180;
Detcen = [Rdet*sin(tthd)*cos(gamd) Rdet*sin(gamd) Rdet*cos(tthd)*cos(gamd)];
Detunit = Detcen./(sqrt(dot(Detcen,Detcen)));
jvec = cross([0 1 0],Detunit); % horizontal pixels unit vector (low pixel number, low tth)
jvec = jvec./(sqrt(dot(jvec,jvec))); %make sure it is a unit vector
ivec = cross(jvec,Detunit); % vertical pixels unit vector (low pixel number, more up)

%Sample 00L rod unit vector
%samvec=[cos(sch)*cos(sth) sin(sch) (-cos(sch)*sin(sth))];
%Sample rotation from phi --> theta, [00L] (sqz) starts outboard at theta 0
sth=Samth*pi/180; sph=Samphi*pi/180; sch=Samchi*pi/180;
sROT1 = [cos(sth) 0 sin(sth); 0 1 0; -sin(sth) 0 cos(sth)];
sROT2 = [cos(sch) -sin(sch) 0; sin(sch) cos(sch) 0; 0 0 1];
sROT3 = [1 0 0; 0 cos(sph) -sin(sph); 0 sin(sph) cos(sph)];
sROT = sROT1*sROT2*sROT3; %nanoprobe stage stack
%sROT = sROT2*sROT1*sROT3; %LCLS stage stack
sqx = sROT*[0 0 1]'; sqy = sROT*[0 1 0]'; sqz = sROT*[1 0 0]';

kfmat=zeros(Ndet,Ndet,4);
kimat=zeros(Ndet,Ndet,4);
prefac=zeros(Ndet,Ndet);
qmat=zeros(Ndet,Ndet,4);

% Assign magnitude and vector q to active detector pixels within q and gamma range
%display('assigning q vectors...');
%Generate pixel arrays (um)
pix_x = repmat((imaxis.*(pxsz)),Ndet,1);
pix_y = repmat((imaxis.*(pxsz))',1,Ndet);
%Calculate real space pixel positions in um as a vector sum
kfmat(:,:,1) = pix_x(:,:).*jvec(1)+pix_y(:,:).*ivec(1)+Detcen(1);
kfmat(:,:,2) = pix_x(:,:).*jvec(2)+pix_y(:,:).*ivec(2)+Detcen(2);
kfmat(:,:,3) = pix_x(:,:).*jvec(3)+pix_y(:,:).*ivec(3)+Detcen(3);
kfmat(:,:,4) = sqrt(kfmat(:,:,1).^2+kfmat(:,:,2).^2+kfmat(:,:,3).^2);
%Assign kfinal unit vector to each pixel based on position
kfmat(:,:,1) = kfmat(:,:,1)./kfmat(:,:,4);
kfmat(:,:,2) = kfmat(:,:,2)./kfmat(:,:,4);
kfmat(:,:,3) = kfmat(:,:,3)./kfmat(:,:,4);
%Find kinitial unit vector symmetric about 00L rod (starts parallel +x)
prefac(:,:) = -2*(kfmat(:,:,1).*sqz(1)+kfmat(:,:,2).*sqz(2)+kfmat(:,:,3).*sqz(3));
kimat(:,:,1) = kfmat(:,:,1)+prefac(:,:).*sqz(1);
kimat(:,:,2) = kfmat(:,:,2)+prefac(:,:).*sqz(2);
kimat(:,:,3) = kfmat(:,:,3)+prefac(:,:).*sqz(3);
%Scale kinitial back to optic plane, find radial intersection in zp
prefac(:,:) = abs(Fzp./kimat(:,:,3)).*sqrt(kimat(:,:,1).*kimat(:,:,1)+kimat(:,:,2).*kimat(:,:,2));
kimat(:,:,4) = (prefac>(Dstop/2)).*(prefac<(Dzp/2));
%Calculate q vectors
qmat(:,:,1) = kb*(kfmat(:,:,1)-kimat(:,:,1));
qmat(:,:,2) = kb*(kfmat(:,:,2)-kimat(:,:,2));
qmat(:,:,3) = kb*(kfmat(:,:,3)-kimat(:,:,3));
qmat(:,:,4) = sqrt(qmat(:,:,1).^2+qmat(:,:,2).^2+qmat(:,:,3).^2);
%Calculate thin film Patterson function per q
imbin = Intensity * ((sin(Num_emit * Refl_d * qmat(:,:,4))).^2)./((sin(Refl_d * qmat(:,:,4))).^2+1e-10);
%Include only pixels within zp angle spread
imbin = imbin.*kimat(:,:,4);

imq=qmat(:,:,4); imqx=qmat(:,:,1);imqy=qmat(:,:,2);imqz=qmat(:,:,3);imki = kimat(:,:,4);

if (plotflag==1)
%     figure(11);set(11,'Position',[0 imb imd imd]);clf;imagesc(imq);axis square;
%     title('Magnitude q per pixel'); pause(0.5);
%     figure(16);set(16,'Position',[imd imb imd imd]);clf;imagesc(imki);axis square;
%     title('zone plate image perfect reflection'); 
% %    figure(17);set(17,'Position',[imd imb imd imd]);clf;imagesc(imqy);axis square;
%    title('Magnitude qy per pixel'); 
%    figure(18);set(18,'Position',[imd imb imd imd]);clf;imagesc(imqz);axis square;
%    title('Magnitude qz per pixel'); 
    figure(102);clf;imagesc(imbin);axis square off;colormap hot;colorbar;%set(19,'Position',[2*imd imb imd imd]);
    title('Theoretical image');%caxis([0 50]);
end



dataout=imbin;
end