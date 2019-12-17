function vertlines(xpts)
    hold on;
    for xval = xpts
        plot([xval xval], ylim, 'k:');
    end
end
