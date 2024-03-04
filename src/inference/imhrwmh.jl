
struct MetropolisHastings{D <: UnivariateDistribution, F <: Real}
    imh_proposal::D
    rwmh_sigma  ::F
    imh_weight  ::F
end

function transition_imh(rng::Random.AbstractRNG, model, q, θ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    ℓw′ = ℓπ′ - logpdf(q, θ′)
    ℓw = ℓπ - logpdf(q, θ)
    α  = exp(ℓw′ - ℓw)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end

function transition_rwmh(rng::Random.AbstractRNG, model, σ, θ::Real)
    q  = Normal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    α  = exp(ℓπ′ - ℓπ)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end

function transition_rwmh(rng::Random.AbstractRNG, model, σ, θ::AbstractVector)
    q  = MvNormal(θ, σ)
    θ′  = rand(rng, q)
    ℓπ′ = logdensity(model, θ′)
    ℓπ = logdensity(model, θ)
    α  = exp(ℓπ′ - ℓπ)
    if rand(rng) < α
        θ′, ℓπ′, α
    else
        θ, ℓπ, α
    end
end
