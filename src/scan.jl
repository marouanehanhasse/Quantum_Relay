using JuMP
function scan_maker(A)
    m = JuMP.Model(solver=ClpSolver(PrimalTolerance=1e-3, DualTolerance=1e-3, InfeasibleReturn=1, PresolveType=1))
    # m = Model(solver=GurobiSolver())
    level = size(A, 2)
    v = zeros(Int, level)
    ub = zeros(Int, level)
    lb = zeros(Int, level)

    @variable(m, x[1:level])
    @constraint(m, con, A*x.>=0)

    function setc(c)
        for i = 1:size(A, 1)
            m.linconstr[i].lb = float(c[i])
        end
    end
    
    function scan(c::Channel)
        i = 1
        init = 1
        while i > 0
            if i >= init
                @objective(m, Max, x[i])
                res = JuMP.solve(m, suppress_warnings=true)
                if res==:Optimal || res==:Unbounded
                    ub[i] = round(Int, getvalue(x[i]))
                    setobjectivesense(m, :Min)
                    res = JuMP.solve(m, suppress_warnings=true)
                    @assert res==:Optimal || res==:Unbounded
                    lb[i] = round(Int, getvalue(x[i]))

                    v[i] = lb[i]
                    init += 1
                else
                    @assert res==:Infeasible
                    i -= 1
                    continue
                end
            elseif v[i] < ub[i]
                v[i] += 1
            else
                setupperbound(x[i], Inf)
                setlowerbound(x[i], -Inf)
                init -= 1
                i -= 1
                continue
            end

            if i >= level
                put!(c, v)
                continue
            else
                setupperbound(x[i], v[i])
                setlowerbound(x[i], v[i])
                i += 1
            end
        end
        close(c)
    end
    
    return setc, scan
end
