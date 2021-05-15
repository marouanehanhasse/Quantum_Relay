using GLPK
using JuMP
function scan_maker(A)
    m= Model(GLPK.Optimizer)

    level = size(A, 2)
    v = zeros(Int, level)
    ub = zeros(Int, level)
    lb = zeros(Int, level)

    @variable(m, x[1:level])
    @constraint(m, con, A*x.>=0)

    function setc(a)
        set_normalized_rhs.(con, float.(a))
    end
    
    function scan(c::Channel)
        i = 1
        init = 1
        while i > 0
            if i >= init
                @objective(m, Max, x[i])
                res = JuMP.optimize!(m)
                if termination_status(m) == MOI.OPTIMAL || termination_status(m) == MOI.DUAL_INFEASIBLE
                    ub[i] = round(Int, getvalue(x[i]))
                    set_objective_sense(m, MOI.MIN_SENSE)
                    res = JuMP.optimize!(m)
                    @assert termination_status(m) == MOI.OPTIMAL || termination_status(m) == MOI.DUAL_INFEASIBLE
                    lb[i] = round(Int, getvalue(x[i]))

                    v[i] = lb[i]
                    init += 1
                else
                    @assert termination_status(m) == MOI.INFEASIBLE
                    i -= 1
                    continue
                end
            elseif v[i] < ub[i]
                v[i] += 1
            else
                set_upper_bound(x[i], Inf)
                set_lower_bound(x[i], -Inf)
                init -= 1
                i -= 1
                continue
            end

            if i >= level
                put!(c, v)
                continue
            else
                set_upper_bound(x[i], v[i])
                set_lower_bound(x[i], v[i])
                i += 1
            end
        end
        close(c)
    end
    
    return setc, scan
end
