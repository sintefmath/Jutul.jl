function data = class2data(obj)
    if(isstruct(obj) || isobject(obj))
        myfields = fields(obj);
        if(numel(obj)>1)
            assert(isstruct(obj))
            data = obj;
            for k=1:numel(obj)  
                for i=1:numel(myfields)
                    data(k).(myfields{i}) = class2data(obj(k).(myfields{i}));            
                end
            end
        else
            data = struct();
            for i=1:numel(myfields)
                    data.(myfields{i}) = class2data(obj.(myfields{i}));            
            end
        end
    else
       if(isnumeric(obj) || ischar(obj) )
                data = obj;
       elseif (iscell(obj))
           vals = obj;
           newvals ={};
           for i=1:numel(vals)
               newvals{i} = class2data(vals{i});
           end
           data = newvals;
       else
           data = [];
       end 
    end
end