
using Test
using WidebandDoA

using AbstractMCMC
using Base.Iterators
using Distributions
using FFTW
using LinearAlgebra
using MCMCTesting
using Random
using ReversibleJump
using StableRNGs
using Statistics
using Tullio

#include("filters.jl")
#include("striped.jl")

function MCMCTesting.sample_joint(
    rng  ::Random.AbstractRNG,
    model::WidebandDoA.WidebandNormalGammaPrior
)
    all_params = WidebandDoA.sample_params(rng, model)
    y          = WidebandDoA.sample_signal(rng, model, all_params)
    params     = @. WidebandDoA.WidebandNormalGammaParam(all_params.phi, log(all_params.lambda))
    params, y
end

include("mcmc.jl")
#include("rjmcmc.jl")
