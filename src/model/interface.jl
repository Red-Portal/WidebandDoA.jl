
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

struct WidebandNormalGammaConstantOrder{M}
    model::M
    order::Int
end

function logdensity(
    wrapper::WidebandNormalGammaConstantOrder,
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
    model_wrapper   = WidebandNormalGammaConstantOrder(model, order)
    ϕ               = T[θj.phi       for θj in θ]
    ℓλ              = T[θj.loglambda for θj in θ]
    θ_flat          = vcat(ϕ, ℓλ)
    θ_flat, ℓp, _   = slice_sampling(rng, sampler_adapted, model_wrapper, θ_flat)
    θ               = WidebandNormalGammaParam{T}[
        WidebandNormalGammaParam(θ_flat[i], θ_flat[order+i]) for i in 1:order
    ]
    θ, ℓp
end

