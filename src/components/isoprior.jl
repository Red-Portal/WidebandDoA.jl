
struct WidebandIsoSourcePrior{
    F  <: Real,
    OP <: DiscreteDistribution,
    SP <: UnivariateDistribution,
} <: AbstractWidebandPrior
    n_samples   ::Int
    n_fft       ::Int
    alpha       ::F
    beta        ::F
    order_prior ::OP
    source_prior::SP
end

function logpriordensity(
    prior::WidebandIsoSourcePrior,
    θ    ::AbstractVector
)
    ϕ      = [θi.phi       for θi in θ]
    ℓλ     = [θi.loglambda for θi in θ]
    λ      = exp.(ℓλ)
    ℓjac_λ = sum(ℓλ)

    @unpack order_prior, source_prior = prior
    k    = length(ϕ)
    ℓp_k = logpdf(order_prior, k)

    if k == 0
        ℓp_k   
    elseif any((ϕ .< -π/2) .|| (ϕ .> π/2))
        -Inf
    else
        ℓp_ϕ = -k*log(π)
        ℓp_λ = sum(Base.Fix1(logpdf, source_prior), λ)
        ℓp_ϕ + ℓp_λ + ℓp_k + ℓjac_λ
    end
end

function Base.rand(
    rng   ::Random.AbstractRNG,
    prior ::WidebandIsoSourcePrior;
    k     ::Int            = rand(rng, prior.order_prior),
    sigma ::Real           = rand(rng, InverseGamma(prior.alpha, prior.beta)),
    phi   ::AbstractVector = rand(rng, Uniform(-π/2, π/2), k),
    lambda::AbstractVector = rand(rng, prior.source_prior, k)
)
    @unpack n_samples, n_fft, alpha, beta, order_prior, source_prior = prior
    z_x = randn(rng, n_fft, k)
    Tullio.@tullio x[n,k] := sqrt(lambda[k])*sigma*z_x[n,k]
    (k=k, phi=phi, lambda=lambda, sigma=sigma, sourcesignals=x)
end
