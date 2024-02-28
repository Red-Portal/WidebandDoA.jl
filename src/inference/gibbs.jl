
struct GibbsObjective{Model, Idx <: Integer, Vec <: AbstractVector}
    model::Model
    idx  ::Idx
    θ    ::Vec
end

function logdensity(obj::GibbsObjective, θi)
    @unpack model, idx, θ = obj
    θ′ = @set θ[idx] = θi
    logdensity(model, θ′)
end
