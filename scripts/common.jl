
using Distributions
using DrWatson
using DSP
using ReversibleJump
using WidebandDoA
using Bootstrap

function construct_default_model(
    rng::Random.AbstractRNG, ϕ::AbstractVector, snr::Real
)
    N      = 32
    M      = 20
    Δx     = range(0, M*0.5; length=M)
    c      = 1500
    fs     = 1000
    
    filter = WidebandDoA.WindowedSinc(N)
    λ      = fill(10^(snr/10), length(ϕ))
    σ      = 1.0

    # P[λ > 0.1] = 95%
    α_λ, β_λ = 2.1, 0.4905160381762056
    α, β     = 0., 0.

    order_prior = NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1))
    model       = WidebandDoA.WidebandNormalGammaPrior(
        N, filter, Δx, c, fs, order_prior, α_λ, β_λ, α, β,
    )

    θ = (k=2, phi=ϕ, lambda=λ, sigma=σ)
    y = WidebandDoA.sample_signal(rng, model, θ)

    model = WidebandDoA.WidebandNormalGamma(
        y, Δx, c, fs, α_λ, β_λ, α, β; delay_filter=filter
    )
    model, θ
end

function run_bootstrap(
    data′;
    sampling_strategy = BalancedSampling(1024),
    confint_strategy  = PercentileConfInt(0.8),
)
    boot = bootstrap(mean, data′, sampling_strategy)
    μ, μ_hi, μ_lo = confint(boot, confint_strategy) |> only
    (μ, μ_hi - μ, μ_lo - μ)
end

function sample_bandlimited_signals(
    rng    ::Random.AbstractRNG,
    prior  ::WidebandDoA.WidebandNormalGammaPrior,
    params ::NamedTuple,
    f_begin::Real,
    f_end  ::Real
)
    @unpack n_snapshots, order_prior, c, Δx, fs, delay_filter = prior
    @unpack k, phi, lambda, sigma = params
    
    n_sensor = length(Δx)
    N, M     = n_snapshots, n_sensor
    ϕ, λ, σ  = phi, lambda, sigma
    k        = length(ϕ)

    bpf = DSP.Filters.digitalfilter(
        DSP.Filters.Bandpass(f_begin, f_end, fs=fs), 
        DSP.Filters.Butterworth(8)
    )
    z_a = randn(rng, N, k)
    a   = mapreduce(hcat, zip(λ, eachcol(z_a))) do (λj, z_aj)
        aj = sqrt(λj)*z_aj
        reshape(DSP.Filters.filt(bpf, aj), (:,1))
    end
    y = WidebandDoA.simulate_propagation(rng, prior, params, a)
    y, a
end
