using Base: run_extension_callbacks

using Accessors
using AbstractMCMC
using Base.Iterators
using Distributions
using DelimitedFiles
using Random, Random123
using ReversibleJump
using Tullio
using WidebandDoA
using ProgressMeter
using Plots

include("common.jl")

function main(snr)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng  = Random123.Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, 1)

    ϕ             = -π/4
    model, θ_true = construct_default_model(rng, [ϕ], snr)
    y, a          = sample_bandlimited_signals(rng, model.prior, θ_true, 100, 200)
    a_std         = std(a)
    y            /= a_std
    a            /= a_std

    Plots.plot(a[:,1], color=:red) |> display

    model = WidebandNormalGamma(y, model.prior)

    n_burn       = 1000
    n_samples    = 1000
    n_thin       = 10

    θ0           = [WidebandDoA.WidebandNormalGammaParam(ϕ, randn(rng))]
    mcmc         = SliceSteppingOut([2.0, 2.0])
    θ            = copy(θ0)
    params_chain = Vector{typeof(θ0)}(undef, n_samples)

    for _ in 1:n_burn
        θ, _, _ = transition_mcmc(rng, mcmc, model, θ)
    end

    for t in 1:n_samples
        θ, _, _ =  transition_mcmc(rng, mcmc, model, θ)
        params_chain[t] = θ
    end

    a_recon_chain = map(params_chain[1:n_thin:end]) do θ
        a_cond = WidebandDoA.reconstruct(model, θ)
        (rand(rng, a_cond), mean(a_cond))
    end
    a_recon_chain_samples = mapreduce(first, hcat, a_recon_chain)
    a_recon_chain_mean    = mean(last, a_recon_chain)

    Plots.plot!(a_recon_chain_mean, c=:blue) |> display
    Plots.plot!(a_recon_chain_samples, alpha=0.1, c=:blue) |> display

    open(datadir("raw", "signal_reconstruction_snr=$(snr).csv"), "w") do io
        x       = (0:size(a,1)-1)
        y_true  = a
        y_mmse  = a_recon_chain_mean
        y_recon = eachcol(a_recon_chain_samples)
        writedlm(io, hcat(x, y_true, y_mmse, y_recon...), ',')
    end
end
