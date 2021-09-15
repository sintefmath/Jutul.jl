function writeOlympusRocks(rocks)
    [f, ~] = fileparts(mfilename('fullpath'));
    savepath = fullfile(f, '..', '..', 'data', 'testgrids');
    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end
    fp = fullfile(savepath, ['olympus_rocks', '.mat']);
    save(fp, 'rocks')
    fprintf('Wrote grid and rock to %s.\n', fp);
end
