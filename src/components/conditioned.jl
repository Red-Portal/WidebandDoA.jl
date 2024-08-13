
struct WidebandData{
    Y  <: AbstractMatrix{<:Real},
    YF <: AbstractMatrix{<:Complex},
    YP <: Real
}
    y        ::Y
    y_fft    ::YF
    y_fft_pad::YF
    y_power  ::YP
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
    y_pad = vcat(y, zeros(size(y,1)-1, size(y,2)))
    Y     = fft(y, 1)     / sqrt(size(y, 1))
    Y_pad = fft(y_pad, 1) / sqrt(size(y_pad, 1))
    P     = sum(abs2, y)
    data  = WidebandData(y, Y, Y_pad, P)
    WidebandConditioned{typeof(model), typeof(data)}(model, data)
end
