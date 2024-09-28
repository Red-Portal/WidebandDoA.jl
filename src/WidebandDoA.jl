
module WidebandDoA

using AbstractMCMC
using Accessors
using Base.Iterators
using DispatchDoctor
using Distributions
using FFTW
using LinearAlgebra
using LoopVectorization
using PDMats
using Random 
using ReversibleJump
using SimpleUnPack
using SparseArrays
using Statistics
using Tullio

@stable default_mode="disable" begin

include("inference/gibbs.jl")
include("inference/slice.jl")
include("inference/imhrwmh.jl")
include("linalg/striped.jl")
include("linalg/schur.jl")
include("linalg/chol.jl")
include("linalg/trsv.jl")

export
    Slice,
    SliceSteppingOut,
    SliceDoublingOut,
    RandomWalkMetropolis,
    IndependentMetropolis,
    MetropolisMixture

abstract type AbstractDelayFilter end

"""
    array_delay(filter, Δn)

Returns the fourier domain fractional delay filters as a matrix
\$H \\in \\mathbb{R}^{ N \\times M \\times K }\$.
The fractional delay filters are the ones in:
"""
function array_delay end

abstract type AbstractWidebandParam end

abstract type AbstractWidebandModel end

abstract type AbstractWidebandConditionedModel <: AbstractMCMC.AbstractModel end

abstract type AbstractWidebandPrior end

"""
    logpriordensity(prior, params)

Log density of `prior` evaluated on `params`.
"""
function logpriordensity end

abstract type AbstractWidebandLikelihood end

"""
    loglikelihood(likelihood, params)

Log likelihood of `likelihood` evaluated on `params`.
"""
function loglikelihood end

function block_fft(m::Int, N::Int)
    idx = 0:N-1
    Tullio.@tullio W[i,j] := exp(-im*2*π*idx[i]/N*idx[j]) / sqrt(N)

    W_sp = sparse(W)
    Φ    = blockdiag(fill(W_sp, m)...)
    Φ, Φ'
end

"""
    reconstruct(cond::WidebandConditioned, params)

Conditional posterior of the latent source signals for reconstruction given conditioned model `cond` and `params`

# Arguments
* `cond::WidebandConditioned`: Conditioned model.
* `params`: Additional parameters we need to condition on.

# Returns
* `cond_post`: Conditional posterior for the latent source signals.
"""
function reconstruct(::WidebandConditioned, ::Any) end


"""
    rand(rng, model)

Sample from a wideband signal model.

# Arguments
* `rng::Random.AbstractRNG`
* `model`: Wideband signal model.

# Returns
* `params`: `NamedTuple` containing the model parameters.
* `data`: Simulated received data.
"""
function Base.rand(::Random.AbstractRNG, ::AbstractWidebandModel) end

export
    WindowedSinc,
    ComplexShift

include("components/filters.jl")

export WidebandConditioned

include("components/conditioned.jl")

export UniformNormalLocalProposal

include("components/localproposals.jl")

export WidebandIsoSourcePrior

include("components/isoprior.jl")

export WidebandIsoIsoLikelihood

include("components/isoisolikelihood.jl")

export
    WidebandIsoIsoModel,
    WidebandIsoIsoParam,
    WidebandIsoIsoMetropolis,
    reconstruct

include("model/isoiso/model.jl")
include("model/isoiso/reconstruct.jl")
include("model/isoiso/rjmcmc_interface.jl")
include("model/isoiso/mcmc_interface.jl")

using ProgressMeter
using StatsFuns

include("relabel.jl")

function __init__()
    @tullio threads=false grad=false fastmath=false
end

end

end
