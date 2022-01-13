name = "runElectrolyte";
path_batman_ex = "../../../../project-batman/Examples/";
run(path_batman_ex + name);
data_folder = "../../../data/models/";
filename = data_folder + "modelElectrolyte" + ".mat";

model = class2data(model);
paramobj = class2data(paramobj);
schedule = class2data(schedule);

save(filename,'model','schedule','state0','paramobj','states')
