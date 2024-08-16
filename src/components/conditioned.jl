
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
    @unpack n_samples, n_fft = model.likelihood

    n_pad = n_fft - n_samples
    y_pad = vcat(y, zeros(n_pad, size(y,2)))
    Y_pad = fft(y_pad, 1) / sqrt(size(y_pad, 1))
    P     = sum(abs2, y)
    data  = WidebandData(y, Y_pad, P)
    WidebandConditioned{typeof(model), typeof(data)}(model, data)
end
