function writeMRSTData(G, rock, filename)
    [f, ~] = fileparts(mfilename('fullpath'));
    savepath = fullfile(f, '..', '..', 'data', 'testgrids');
    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end
    fp = fullfile(savepath, [filename, '.mat']);
    save(fp, 'G', 'rock')
    fprintf('Wrote grid and rock to %s.\n', fp);
end
