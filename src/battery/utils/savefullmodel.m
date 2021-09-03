% run some versjon of test_euler.m
name = 'test_euler.m';
path_batman_ex = "../../../../project-batman/Examples/";
run(path_batman_ex + name);

filename = '../../../data/models/model1d_notemp';
model = class2data(model);
save(filename, 'model', 'state0', 'schedule', 'states')

% can be loaded in julia by MAT exported = MAT.matread('model1d.mat')
