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

function run_reconstruction(rng, ϕ, snr, N, visualize=false)
    ϕ     = [ϕ]
    n_dft = 1024
    fs    = 3000.
    model = construct_default_model(N, fs)
    c, Δx = model.likelihood.c, model.likelihood.Δx

    f_begin   = 10
    f_end     = 1000
    y, a_true = simulate_signal(
        rng, N, n_dft, ϕ, snr, f_begin, f_end, fs, 1.0, Δx, c; visualize
    )

    gain    = sqrt(10^(-snr/10))
    y      *= gain
    a_true *= gain
    cond    = WidebandConditioned(model, y)

    n_burn       = 1000
    n_samples    = 1000
    n_thin       = 10

    θ0           = [WidebandDoA.WidebandIsoIsoParam(only(ϕ), randn(rng))]
    mcmc         = SliceSteppingOut([2.0, 2.0])
    θ            = copy(θ0)
    params_chain = Vector{typeof(θ0)}(undef, n_samples)

    for _ in 1:n_burn
        θ, _, _ = transition_mcmc(rng, mcmc, cond, θ)
    end

    for t in 1:n_samples
        θ, _, _ =  transition_mcmc(rng, mcmc, cond, θ)
        params_chain[t] = θ
    end

    a_recon_chain = map(params_chain[1:n_thin:end]) do θ
        a_cond = WidebandDoA.reconstruct(cond, θ)
        (rand(rng, a_cond)[1:N,:], mean(a_cond)[1:N,:])
    end
    a_recon_chain_samples = mapreduce(first, hcat, a_recon_chain)
    a_recon_chain_mean    = mean(last, a_recon_chain)

    t = (0:size(a_true,1)-1)/fs*1000

    t, a_recon_chain_samples, a_recon_chain_mean, a_true
end

function run_visualization(snr)
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng  = Random123.Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, 1)

    N = 64

    t, a_recon_samples, a_recon_mean, a_true = run_reconstruction(rng, -π/4, snr, N, true)
    Plots.plot(a_true[:,1], color=:red) |> display
    Plots.plot!(a_recon_mean, c=:blue) |> display
    Plots.plot!(a_recon_samples, alpha=0.1, c=:blue, legend=false) |> display

    open(datadir("raw", "signal_reconstruction_snr=$(snr).csv"), "w") do io
        y_true  = a_true[:,1]
        y_mmse  = a_recon_mean
        y_recon = eachcol(a_recon_samples)
        writedlm(io, hcat(0:length(t)-1, y_true, y_mmse, y_recon...), ',')
    end
end

function run_mmse_estimaton()
    n_reps = 512
    for N in [32, 64, 128], ϕ in [0.0, 5/11*π]
        snrs = -10:2:10
        res  = map(snrs) do snr
            mse_estimates = @showprogress pmap(1:n_reps) do key
                seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
                rng  = Random123.Philox4x(UInt64, seed, 8)
                Random123.set_counter!(rng, key)

                _, _, a_recon_mean, a_true = run_reconstruction(rng, ϕ, snr, N, false)
                mean(abs2, a_recon_mean - a_true)
            end
            res       = run_bootstrap(mse_estimates; stat=x -> sqrt(mean(x)))
            est       = res[1]
            est_minus = abs(res[2])
            est_plus  = res[3]
            @info("", N, snr, rmse="$(est) ($(est-est_minus),$(est+est_plus))")
            [est, est_minus, est_plus]
        end

        open(datadir("raw", "signal_reconstruction_rmse_N=$(N)_phi=$(ϕ).csv"), "w") do io
            mmse = Array(hcat(res...)')
            writedlm(io, hcat(snrs, mmse), ',')
        end
    end
end
