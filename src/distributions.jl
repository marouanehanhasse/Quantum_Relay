using Distributions
struct OrthoNNDist <: DiscreteMultivariateDistribution
	x0::Vector{Float64}
	oc::Array{Float64,2}
	x1s::Array
	prob::Float64
    #return a new uniform distribution with all vectors in x1s orthogonal to oc
	function OrthoNNDist(x0::Vector{Float64}, oc::Array{Float64,2})
		x1s = []
		for i = 1:size(oc)[2]
			x1 = x0 + oc[:, i]
			if nonneg(x1)
				push!(x1s, x1)
			end
			x1 = x0 - oc[:, i]
			if nonneg(x1)
				push!(x1s, x1)
			end
		end
		new(x0, oc, x1s, 1.0/length(x1s))
	end
end

Base.length(d::OrthoNNDist) = length(d.x0)

Distributions.rand(d::OrthoNNDist) = rand(d.x1s)

Distributions.pdf(d::OrthoNNDist, x::Vector) = x in d.x1s ? d.prob : 0.0
Distributions.pdf(d::OrthoNNDist) = fill(d.prob, size(d.x1s))
Distributions.logpdf(d::OrthoNNDist, x::Vector) = log(pdf(d, x))
