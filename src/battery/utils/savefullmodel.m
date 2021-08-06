# run some versjon of test_euler.m
filename = 'modelElectrolyte.mat'
%run test_euler.m;filename = 'model1d'
run runElectolyte.m;filename = 'modelElectrolyte.mat'
model = class2data(model)
paramobj = class2data(paramobj)
schedule = class2data(schedule)
save(filename,'model','schedule','state0','paramobj','states')
# can be loaded in julia by MAT exported = MAT.matread('model1d.mat')