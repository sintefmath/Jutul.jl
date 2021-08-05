# run some versjon of test_euler.m
run test_euler.m
model1d=class2data(model)
fn = 'model1d'
save('model1d.mat','model1d')
# can be loaded in julia by MAT exported = MAT.matread('model1d.mat')