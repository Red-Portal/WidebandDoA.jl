
struct WidebandIsoIsoModel{
    Prior <: WidebandIsoSourcePrior,
    Like  <: WidebandIsoIsoLikelihood
} <: AbstractWidebandModel
    prior     ::Prior
    likelihood::Like
end

struct WidebandIsoIsoParam{T <: Real}
    phi::T
    loglambda::T
end

function WidebandIsoIsoModel(
    n_samples   ::Int,
    Δx          ::AbstractVector,
    c           ::Real,
    fs          ::Real,
    source_prior::UnivariateDistribution,
    α           ::Real = 0,
    β           ::Real = 0;
    order_prior ::DiscreteDistribution = NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)),
    n_fft       ::Int  = n_samples*2,
)
    delay_filter = WindowedSinc(n_fft)
    prior = WidebandIsoSourcePrior(
        n_samples,
        n_fft,
        α, β,
        order_prior,
        source_prior
    )
    likelihood = WidebandIsoIsoLikelihood(
        n_samples,
        n_fft,
        delay_filter,
        Δx, c, fs,
    )
    WidebandIsoIsoModel{typeof(prior), typeof(likelihood)}(
        prior, likelihood
    )
end

function ReversibleJump.logdensity(
    target::WidebandConditioned{<: WidebandIsoIsoModel, D},
    θ     ::AbstractVector{<:WidebandIsoIsoParam},
) where {D}
    @unpack data, model       = target
    @unpack likelihood, prior = model

    ℓp_θ = logpriordensity(prior, θ)

    if isfinite(ℓp_θ)
        ℓp_y = loglikelihood(likelihood, prior, data, θ)
        ℓp_y + ℓp_θ
    else
        -Inf
    end
end

function Base.rand(rng::Random.AbstractRNG, model::WidebandIsoIsoModel)
    params = rand(rng, model.prior)
    data   = rand(
        rng,
        model.likelihood,
        params.sourcesignals,
        params.phi;
        sigma=params.sigma
    )
    params, data
end
