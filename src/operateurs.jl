#in this file, op_a and op_b are a,b operators in Barry's original paper.
#H,V means horizontal and vertical
#p means '(prime) for output modes

using SymPy
using Printf
using PyCall
struct QRelaySym{T<:Tuple}
    aH::T
    bH::T
    aV::T
    bV::T
    apH::T
    bpH::T
    apV::T
    bpV::T
end


function trans(op, a, b, ap, bp, B)
    p = B * [ap; bp]
    op = subs(op, a, p[1])
    op = subs(op, b, p[2])
    return op
end

#2D rotation matrix
function rotmat(theta)
    c = cos(theta)
    s = sin(theta)
    return [c s; -s c]
end


function qrelay_op(n, phi, alpha, delta)
    #operators before BS
    aH = symbols(@sprintf("a_H1:%d", n+1))
    bH = symbols(@sprintf("b_H1:%d", n+1))
    aV = symbols(@sprintf("a_V1:%d", n+1))
    bV = symbols(@sprintf("b_V1:%d", n+1))
    
    op = 0
    for i=1:n
        op += phi[i] * (aH[i]*bH[i] + aV[i]*bV[i])
    end

    #beam splitter transformation
    B = 1/sqrt(2)*[1 1;-1 1]
    
    #operators after beam splitter
    apH = symbols(@sprintf("a'_H1:%d", n+1))
    bpH = symbols(@sprintf("b'_H1:%d", n+1))
    apV = symbols(@sprintf("a'_V1:%d", n+1))
    bpV = symbols(@sprintf("b'_V1:%d", n+1))

    for i=1:n-1
        op = trans(op, bH[i], aH[i+1], bpH[i], apH[i+1], B)
        op = trans(op, bV[i], aV[i+1], bpV[i], apV[i+1], B)
    end

    op = trans(op, aH[1], aV[1], apH[1], apV[1], rotmat(alpha))
    op = trans(op, bH[n], bV[n], bpH[n], bpV[n], rotmat(delta))
    
    syms = QRelaySym(aH, bH, aV, bV, apH, bpH, apV, bpV)
    
    return syms, op
end

function op_mat(op)
    op = op.as_poly(domain="C")
    op_a = op.x.gens
    nab = op.length()
    op_ab = ones(SymPy.Sym, nab)
    coef = zeros(Complex, nab)
    mat = zeros(Int64, length(op_a), nab)
    for (i, (ps, c)) in enumerate(op.as_dict())
        for (j, p) in enumerate(ps)
            mat[j, i] = p
            op_ab[i] = op_a[j]^p * op_ab[i]
        end
        coef[i] = c
    end
   
    return op_a, op_ab, mat, coef
end
