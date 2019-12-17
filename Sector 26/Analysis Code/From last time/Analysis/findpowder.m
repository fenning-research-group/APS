function f = findpowder(x,plotflag)
% to fit use the command line
%x=fminsearch(@(x) findpowder(x,0),[169510,6500,-10046,35.2028]);
% to view use
%findpowder(x,1);
% x= 1e5*[1.694031351922207   0.067994415098470  -0.099616200103855
% 0.000352901556651]

%f1 = qbin_pilatus('Images/image275526.tif', x(4), x(1), 0, x(2), x(3));
%f1 = qbin_pilatus('Images/image_000011.tif', x(4), x(1), 0, x(2), x(3), 0, 0, 0, plotflag);
f1 = qbin_pilatus('Images/image_000010.tif', x(4), x(1), 0, x(2), x(3), 0, 0, 0, 0);
if(plotflag)
    figure(701);imagesc(f1.powder);
    figure(702);imagesc(f1.theory);
end
% two theta, R, (gam), X, Y
% input - R X Y 2th
f = f1.error;
%f=f1;
end