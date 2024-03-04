
struct WidebandNormalGammaParam{F <: Real}
    phi      ::F
    loglambda::F
end

struct UniformNormalLocalProposal{F <: Real}
    mu   ::F
    sigma::F
end

function ReversibleJump.local_proposal_sample(
    rng  ::Random.AbstractRNG,
         ::WidebandNormalGamma,
    prop ::UniformNormalLocalProposal
)
    WidebandNormalGammaParam(
        rand(rng, Uniform(-π/2, π/2)),
        rand(rng, Normal(prop.mu, prop.sigma))
    )
end

function ReversibleJump.local_proposal_logpdf(
         ::WidebandNormalGamma,
    prop ::UniformNormalLocalProposal,
    θ    ::AbstractVector{<:WidebandNormalGammaParam},
    j    ::Integer
)
    ℓλ = θ[j].loglambda
    -log(π) + logpdf(Normal(prop.mu, prop.sigma), ℓλ)
end

function ReversibleJump.local_insert(
      ::WidebandNormalGamma,
    θ ::AbstractVector{<:WidebandNormalGammaParam},
    j ::Integer,
    θj::WidebandNormalGammaParam)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(
     ::WidebandNormalGamma,
    θ::AbstractVector{<:WidebandNormalGammaParam},
    j::Integer
)
    deleteat!(copy(θ), j), θ[j]
end

struct WidebandNormalGammaFlat{M}
    model::M
    order::Int
end

function logdensity(
    wrapper::WidebandNormalGammaFlat,
    θ      ::AbstractVector{T}
) where {T <: Real}
    model  = wrapper.model
    order  = length(θ) ÷ 2
    params = [WidebandNormalGammaParam{T}(θ[i], θ[order+i]) for i in 1:order]
    ReversibleJump.logdensity(model, params)
end

function ReversibleJump.transition_mcmc(
    rng    ::Random.AbstractRNG,
    sampler::AbstractSliceSampling,
    model,
    θ      ::AbstractVector{WidebandNormalGammaParam{T}}
) where {T<:Real}
    window_base     = sampler.window
    order           = length(θ)
    window          = vcat(
        fill(first(window_base), order),
        fill(last(window_base), order)
    )
    sampler_adapted = @set sampler.window = window
    model_wrapper   = WidebandNormalGammaFlat(model, order)
    ϕ               = T[θj.phi       for θj in θ]
    ℓλ              = T[θj.loglambda for θj in θ]
    θ_flat          = vcat(ϕ, ℓλ)

    θ_flat, ℓp, acc_rate = slice_sampling(rng, sampler_adapted, model_wrapper, θ_flat)

    θ = WidebandNormalGammaParam{T}[
        WidebandNormalGammaParam(θ_flat[i], θ_flat[order+i]) for i in 1:order
    ]
    θ, ℓp, (mcmc_acceptance_rate=acc_rate,)
end

struct WidebandNormalGammaMetropolis{
    LK <: AbstractMetropolis, PK <: AbstractMetropolis
} <: AbstractMetropolis
    phi_kernel      ::PK
    loglambda_kernel::LK
end

function ReversibleJump.transition_mcmc(
    rng    ::Random.AbstractRNG,
    sampler::WidebandNormalGammaMetropolis,
    model,
    θ      ::AbstractVector{WidebandNormalGammaParam{T}}
) where {T<:Real}
    SimpleUnPack.@unpack phi_kernel, loglambda_kernel = sampler

    order         = length(θ)
    model_wrapper = WidebandNormalGammaFlat(model, order)

    ϕ       = T[θj.phi       for θj in θ]
    ℓλ      = T[θj.loglambda for θj in θ]
    θ_flat  = vcat(ϕ, ℓλ)
    ϕ_range = 1:length(ϕ)
    ℓλ_range = length(ϕ)+1:length(θ_flat)

    ℓp = logdensity(model_wrapper, θ_flat)

    α_ϕ_avg = 0.0
    for (j, ϕ_idx) in enumerate(ϕ_range)
        model_gibbs = GibbsObjective(model_wrapper, ϕ_idx, θ_flat)
        θ′idx, ℓp, α = transition_mh(rng, phi_kernel, model_gibbs, θ_flat[ϕ_idx])
        α_ϕ_avg     = α_ϕ_avg*(j-1)/j + α/j
        θ_flat[ϕ_idx] = θ′idx
    end

    α_ℓλ_avg = 0.0
    for (j, ℓλ_idx) in enumerate(ℓλ_range)
        model_gibbs = GibbsObjective(model_wrapper, ℓλ_idx, θ_flat)
        θ′idx, ℓp, α = transition_mh(rng, loglambda_kernel, model_gibbs, θ_flat[ℓλ_idx])
        α_ℓλ_avg    = α_ℓλ_avg*(j-1)/j + α/j
        θ_flat[ℓλ_idx] = θ′idx
    end
    θ = WidebandNormalGammaParam{T}[
        WidebandNormalGammaParam(θ_flat[i], θ_flat[order+i]) for i in 1:order
    ]
    stats = (phi_acceptance_rate=α_ϕ_avg, loglambda_acceptance_rate=α_ℓλ_avg)
    θ, ℓp, stats
end

