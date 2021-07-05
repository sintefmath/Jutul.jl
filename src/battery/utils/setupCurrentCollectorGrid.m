clear all
close all

nx = 10;
ny = 10;

Lx = 1;
Ly = 1;

G = cartGrid([nx, ny], [Lx, Ly]);
G = computeGeometry(G);

nc = G.cells.num;

rock.perm = 0.1*darcy*ones(nc, 1);

epsi = Lx/(10*nx);
bcfaces = find(abs(G.faces.centroids(:, 1)) < epsi);
bccells = sum(G.faces.neighbors(bcfaces, :), 2);

figure
plotGrid(G)
plotFaces(G, bcfaces, 'edgecolor', 'red', 'linewidth', 3);
plotGrid(G, bccells, 'facecolor', 'blue');

savedir = '../../../data/testgrids';

save(fullfile(savedir, 'cccase.mat'), 'G', 'rock');
save(fullfile(savedir, 'bccccase.mat'), 'bccells', 'bcfaces');
