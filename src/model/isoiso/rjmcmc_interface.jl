
function ReversibleJump.local_proposal_sample(
    rng  ::Random.AbstractRNG,
         ::WidebandConditioned{<: WidebandIsoIsoModel, D},
    prop ::UniformNormalLocalProposal
) where {D}
    WidebandIsoIsoParam(
        rand(rng, Uniform(-π/2, π/2)),
        rand(rng, Normal(prop.mu, prop.sigma))
    )
end

function ReversibleJump.local_proposal_logpdf(
         ::WidebandConditioned{<: WidebandIsoIsoModel, D},
    prop ::UniformNormalLocalProposal,
    θ    ::AbstractVector{<:WidebandIsoIsoParam},
    j    ::Integer
) where {D}
    ℓλ = θ[j].loglambda
    -log(π) + logpdf(Normal(prop.mu, prop.sigma), ℓλ)
end

function ReversibleJump.local_insert(
      ::WidebandConditioned{<: WidebandIsoIsoModel, D},
    θ ::AbstractVector{<:WidebandIsoIsoParam},
    j ::Integer,
    θj::WidebandIsoIsoParam
) where {D}
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(
     ::WidebandConditioned{<: WidebandIsoIsoModel, D},
    θ::AbstractVector,
    j::Integer
) where {D}
    deleteat!(copy(θ), j), θ[j]
end

