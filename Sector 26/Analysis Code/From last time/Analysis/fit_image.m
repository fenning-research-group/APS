function f = fit_image(x,img_exp,thickness,plotflag)
%Function for finding best fit strain and rotation
%generates theoretical image from sample angles, d spacing, and intensity
%returns error between experimental and theoretical image

%example syntax:
%>>im1 = double(imread('Images/184/scan_184_img_056673.tif'));
%>>x1 = [13.7545,1.2459,5.7570,0.2720];
%>>fit_image(x1,im1,.015,1)
%>>clear x
%>>x=fminsearch(@(x) fit_image(x,im1,.015,0),x1);
%>>fit_image(x1,im1,1)
%>>im2 = double(imread('Images/184/scan_184_img_059310.tif'));
%>>x2 = [13.7109 -0.5215 5.8431 0.0224];
%>>fit_image(x2,im2,.045,1);

im2 = theory_image(37.46,8.7e5,0,x(1),x(2),0,thickness,x(3),10.0,x(4),plotflag);

if(plotflag)
    figure(101);imagesc(img_exp);title('Experimental image');colormap hot;colorbar;axis square tight off
    caxis([0 50]);
end

f = sum(sum((img_exp - im2).^2));