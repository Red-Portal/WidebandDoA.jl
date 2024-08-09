
struct WidebandData{
    Y  <: AbstractMatrix{<:Real},
    YF <: AbstractMatrix{<:Complex},
    YP <: Real
}
    y      ::Y
    y_fft  ::YF
    y_power::YP
end

struct WidebandConditioned{
    M <: AbstractWidebandModel,
    D <: WidebandData
} <: AbstractWidebandConditionedModel
    model::M
    data ::D
end

function WidebandConditioned(
    model::AbstractWidebandModel,
    y    ::AbstractMatrix{<:Real},
)
    Y     = fft(y, 1) / sqrt(size(y, 1))
    P     = sum(abs2, y)
    data  = WidebandData(y, Y, P)
    WidebandConditioned{typeof(model), typeof(data)}(model, data)
end
