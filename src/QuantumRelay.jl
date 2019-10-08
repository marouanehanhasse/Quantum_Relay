module QuantumRelay

using SymPy
using JuMP
using Clp
using IterTools
using LLLplus
using GSL
using PyPlot
using Mamba
using Distributions
using LinearAlgebra
using SmithNormalForm

export
	qrelay_op,
	op_mat,
	scan_maker,
	QRelaySampler

include("utility.jl")
include("operateurs.jl")
include("scan.jl")
include("distributions.jl")

struct QRelaySampler
	prob::Function #return the probablility
	psetproposal::Function #return next combination of sample

    #input parameters:
    #mat: the matrix p_ij in the note (10)
    #coef: the coefficient c in the note (10)
    #omega: the coefficient A in the note (10)
    #pdet0: the probability of detection for each detector
	function QRelaySampler(mat::Array{T, 2}, coef, omega, pdet0) where T <: Int
		B=smith(mat)
        U=B.S
        V=F.T
        S=diagm(F)
        #the PolyLib returns P=USV. Inverse the matrices so Ui/Vi is the same as U/V in the note (18)
		Ui = inv(U) 
		Vi = inv(V)
		s = diag(S)
		r = countnz(s)
		s0 = s[1:r]
		@assert s0 == ones(r)
		ui1 = Ui[1:r, :]
		ui2 = Ui[r+1:end, :]
		vi1 = Vi[:, 1:r]
		vi2 = Vi[:, r+1:end]
		vi2 = lll(vi2)[1] #Lenstra–Lenstra–Lovász lattice basis reduction
		T0 = vi1*ui1
		ui2oc = orthocomp(ui2) #orhogonal complement
		setc, scan = scan_maker(vi2) #make the scanner for the algorithm1 in the note
        
        #compute the probability for an ideal system
        #na: the photon numbers in a output mode
		function prob(na)
		    @assert countnz(ui2*na) == 0
		    b = T0*na
		    setc(-b)
		    total = 0.0
		    for x in Task(scan)
		        nab = vi2*x + b #the photon numbers for each item in the sum in the note (10)
		        total += prod([c.^complex(n)/factorial(n) for (c, n) in zip(coef, nab)])
		    end
		    return abs(total*omega)^2
		end

        #compute the probability of detection
        #q: the number of photons detectors report
        #na: the number of photons arrived at detector
        #mask: if there is no detector in this channel, mask=0
		function prob(q, na, mask)
		    q0 = round(Int, q.>0)
		    m0 = round(Int, mask)
		    return prod((q0 + (1-2q0).*pdet0(na)).^m0)
		end

		psetproposal(x::Vector) = QuantumRelay.OrthoNNDist(x, ui2oc)

		new(prob, psetproposal)
	end
end

end
