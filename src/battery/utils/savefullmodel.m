% run some versjon of test_euler.m
% run test_euler.m;
% filename = '../../../data/models/model1d';

run ../../../../project-batman/Examples/runElectrolyte.m
filename = '../../../data/models/modelElectrolyte.mat';
model = class2data(model);
paramobj = class2data(paramobj);
schedule = class2data(schedule);

% save(filename, 'model', 'states')
save(filename,'model','schedule','state0','paramobj','states')
% can be loaded in julia by MAT exported = MAT.matread('model1d.mat')