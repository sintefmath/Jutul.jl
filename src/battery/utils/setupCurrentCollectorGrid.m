mrstModule add ad-core battery mpfa

nx = 3;
ny = 3;

Lx = 1e-4;
Ly = 1e-4;

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

save(fullfile(savedir, 'square_current_collector.mat'), 'G', 'rock');
save(fullfile(savedir, 'square_current_collector_T.mat'), 'bccells', 'T');
save(fullfile(savedir, 'square_current_collector_P.mat'), 'P', 'S');
