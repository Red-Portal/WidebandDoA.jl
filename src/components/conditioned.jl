

"""
    WidebandData(y, y_fft, y_power)

Received signal with pre-processing.

# Arguments
* `y`: Received signal, where the rows are the channels (sersors), while the columns are the signals.
* `y_fft`: Received signal after applying a channel-wise FFT.
* `y_power`: Power of the received signal.
"""
struct WidebandData{
    Y  <: AbstractMatrix{<:Real},
    YF <: AbstractMatrix{<:Complex},
    YP <: Real
}
    y      ::Y
    y_fft  ::YF
    y_power::YP
end

"""
    WidebandConditioned(model, y)

`model` conditioned on `y`.

# Arguments
* `model::AbstractWidebandModel`: Signal model.
* `y::AbstractMatrix`: Received data, where the rows are the channels (sersors), while the columns are the signals.
"""
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
