mrstModule add ad-core battery mpfa

nx = 10;
ny = 10;

Lx = 1;
Ly = 1;

G = cartGrid([nx, ny], [Lx, Ly]);
G = computeGeometry(G);

nc = G.cells.num;

rock.perm = ones(nc, 1);

epsi = Lx/(10*nx);
bcfaces = find(abs(G.faces.centroids(:, 1)) < epsi);
bccells = sum(G.faces.neighbors(bcfaces, :), 2);

T = computeTrans(G, rock);

nf = G.faces.num;
ncf = size(G.cells.faces, 1);

M = sparse(G.cells.faces(:, 1), (1 : ncf)', 1, nf, ncf);

T = M*T;

T = T(bcfaces);

paramobj = CurrentCollectorInputParams();

paramobj.G = G;
paramobj.heatCapacity = 0;
paramobj.thermalConductivity = 0;
paramobj.heatCapacity = 0;
paramobj.EffectiveElectricalConductivity = 1;


model = CurrentCollector(paramobj);
op = model.operators.cellFluxOp;
P = op.P;
S = op.S;

figure
plotGrid(G)
plotFaces(G, bcfaces, 'edgecolor', 'red', 'linewidth', 3);
plotGrid(G, bccells, 'facecolor', 'blue');

savedir = '../../../data/testgrids';

name = "square_current_collector_10by10";
save(fullfile(savedir, name + '.mat'), 'G', 'rock');
save(fullfile(savedir, name + '_T.mat'), 'bccells', 'T');
save(fullfile(savedir, name + '_P.mat'), 'P', 'S');
