
using Distributions
using DrWatson
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

    # P[λ > 0.1] 80%
    α_λ, β_λ = 2.1, 0.3125930624954082
    α, β     = 0., 0.

    order_prior = NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1))
    model       = WidebandDoA.WidebandNormalGammaPrior(
        N, filter, Δx, c, fs, order_prior, α_λ, β_λ, α, β,
    )

    θ = (k=2, phi=ϕ, lambda=λ, sigma=σ)
    y = WidebandDoA.sample_signal(rng, model, θ)

    WidebandDoA.WidebandNormalGamma(
        y, Δx, c, fs, α_λ, β_λ, α, β; delay_filter=filter
    )
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
