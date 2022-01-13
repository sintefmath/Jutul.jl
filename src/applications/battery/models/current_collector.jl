export CurrentCollector

struct CurrentCollector <: ElectroChemicalComponent end

function minimum_output_variables(
    system::CurrentCollector, primary_variables
    )
    [:TPkGrad_Phi, :Charge]
end

function select_primary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:Phi] = Phi()
end

function select_secondary_variables_system!(
    S, domain, system::CurrentCollector, formulation
    )
    S[:TPkGrad_Phi] = TPkGrad{Phi}()
    S[:Charge] = Charge()
    S[:Conductivity] = Conductivity()
end

function select_equations_system!(
    eqs, domain, system::CurrentCollector, formulation    )
    charge_cons = (arg...; kwarg...) -> Conservation(Charge(), arg...; kwarg...)
    eqs[:charge_conservation] = (charge_cons, 1)
end

function sineup(y1, y2, x1, x2, x)
    #SINEUP Creates a sine ramp function
    #
    #   res = sineup(y1, y2, x1, x2, x) creates a sine ramp
    #   starting at value y1 at point x1 and ramping to value y2 at
    #   point x2 over the vector x.
        
        dy = y1 - y2; 
        dx = abs(x1 - x2);
        res = 0.0 
         if  (x >= x1) && (x <= x2)
            res = dy/2.0.*cos(pi.*(x - x1)./dx) + y1 - (dy/2) 
        end
        
        if     (x > x2)
            res .+= y2
        end

        if  (x < x1)
            res .+= y1
        end
        return res
    
end

function apply_forces_to_equation!(storage, 
    model::SimulationModel{<:Any, <:CurrentCollector, <:Any, <:Any},
    law::Conservation{Charge}, force, time)
    cell = force.cell
    rate = force.src
    tup = 0.1
    inputI = 9.4575
    #equation = get_entries(eq)
    acc = get_entries(law.accumulation)
    t = time
    if ( t<= tup)
        val = sineup(0, inputI, 0, tup, t) 
    else
        val = inputI;
    end
    #val = (t <= tup) .* sineup(0, inputI, 0, tup, t) + (t > tup) .* inputI;
    acc[cell] -= val
    #for cell in cells
    #    equation[cell] += rate
    #end
end