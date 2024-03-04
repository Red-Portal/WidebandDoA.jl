
module WidebandDoA

using AbstractMCMC
using Accessors
using Base
using Distributions
using FFTW
using LinearAlgebra
using Random 
using ReversibleJump
using SimpleUnPack
using Statistics

using LoopVectorization
using Tullio

include("inference/gibbs.jl")
include("inference/slice.jl")
include("inference/imhrwmh.jl")
include("linalg/striped_matrix.jl")

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

abstract type AbstractWidebandModel <: AbstractMCMC.AbstractModel end

include("model/model.jl")
include("model/filters.jl")
include("model/interface.jl")
include("model/likelihood.jl")
include("model/sampling.jl")

export
    WidebandNormalGamma,
    UniformNormalLocalProposal, 
    WidebandNormalGammaMetropolis

end
