function writeMRSTData(G, rock, filename, W)
    if nargin < 4
        W = [];
    end
    W = applyFunction(@(x) x, W);
    [f, ~] = fileparts(mfilename('fullpath'));
    savepath = fullfile(f, '..', '..', 'data', 'testgrids');
    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end
    fp = fullfile(savepath, [filename, '.mat']);
    save(fp, 'G', 'rock', 'W')
    fprintf('Wrote grid and rock to %s.\n', fp);
end
