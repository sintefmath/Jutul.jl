% run some versjon of test_euler.m
run test_euler.m;
filename = '../../../data/models/model1d';
model = class2data(model);
save(filename, 'model')
% can be loaded in julia by MAT exported = MAT.matread('model1d.mat')