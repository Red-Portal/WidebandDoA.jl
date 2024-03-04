
abstract type AbstractMetropolis <: AbstractMCMC.AbstractSampler end

struct RandomWalkMetropolis{F <: Real} <: AbstractMetropolis
    sigma::F
end

struct IndependentMetropolis{D <: UnivariateDistribution} <: AbstractMetropolis
    proposal::D
end

struct MetropolisMixture{
    F    <: Real,
    IMH  <: IndependentMetropolis,
    RWMH <: RandomWalkMetropolis,
} <: AbstractMetropolis
    imh_weight::F
    imh       ::IMH
    rwmh      ::RWMH

    function MetropolisMixture(
        imh       ::IndependentMetropolis,
        rwmh      ::RandomWalkMetropolis,
        imh_weight::Real = 0.2,
    )
        @assert 0 ≤ imh_weight ≤ 1
        new{typeof(imh_weight), typeof(imh), typeof(rwmh)}(imh_weight, imh, rwmh)
    end
end

function transition_mh(rng::Random.AbstractRNG, kernel::MetropolisMixture, model, θ)
    if rand(rng, Bernoulli(kernel.imh_weight))
        transition_mh(rng, kernel.imh, model, θ)
    else
        transition_mh(rng, kernel.rwmh, model, θ)
    end
end

function transition_mh(
    rng::Random.AbstractRNG, kernel::IndependentMetropolis, model, θ
)
    q  = kernel.proposal
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    ℓw′ = ℓπ′ - logpdf(q, θ′)
    ℓw = ℓπ - logpdf(q, θ)
    α  = min(exp(ℓw′ - ℓw), 1)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end

function transition_mh(
    rng::Random.AbstractRNG, kernel::RandomWalkMetropolis, model, θ::Real
)
    σ  = kernel.sigma
    q  = Normal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    α  = min(exp(ℓπ′ - ℓπ), 1)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end

function transition_mh(
    rng::Random.AbstractRNG, kernel::RandomWalkMetropolis, model, θ::AbstractVector
)
    σ  = kernel.sigma
    q  = MvNormal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    α  = min(exp(ℓπ′ - ℓπ), 1)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end
