
using Test
using WidebandDoA

using AbstractMCMC
using Base.Iterators
using Distributions
using FFTW
using LinearAlgebra
using MCMCTesting
using Preferences
using Random
using ReversibleJump
using StableRNGs
using Statistics
using Tullio

set_preferences!("WidebandDoA", "instability_check" => "error")

include("filters.jl")
include("striped.jl")

function MCMCTesting.sample_joint(
    rng  ::Random.AbstractRNG,
    model::WidebandDoA.WidebandIsoIsoModel,
)
    params, data = rand(rng, model)
    params_struct = @. WidebandIsoIsoParam(params.phi, log(params.lambda))
    params_struct, data
end

include("mcmc.jl")
include("rjmcmc.jl")
