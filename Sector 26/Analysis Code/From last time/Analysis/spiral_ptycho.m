Npts=401; %scan points
Xstart=0.0; %microns
Ystart=0.0;  %microns

scale_val_X = .03;
scale_val_Y = .03;

positions=zeros(Npts,2);
for ii=0:Npts-1
   positions(ii+1,1)=Xstart + scale_val_X*sqrt(ii)*cos(4*sqrt(ii));
   positions(ii+1,2)=Ystart + scale_val_Y*sqrt(ii)*sin(4*sqrt(ii));
end
figure(54);clf reset;
plot(positions(:,1),positions(:,2),'-o','LineWidth',2,'MarkerSize',10);
axis square tight;
FOVX=max(max(positions(:,1)))-min(min(positions(:,1)));
FOVY=max(max(positions(:,2)))-min(min(positions(:,2)));
dists=sqrt((positions(1:(Npts-1),1)-positions(2:Npts,1)).^2+...
    (positions(1:(Npts-1),2)-positions(2:Npts,2)).^2);
display(['Field of view in X(nm):' num2str(FOVX*1000) '  FOV in Y(nm):' num2str(FOVY*1000) ...
    '   Point separation (nm)' num2str(mean(mean(dists))*1000)]);
