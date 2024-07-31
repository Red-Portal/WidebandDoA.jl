
struct ArrayConfig{F <: Real, V <: AbstractVector{<:Real}}
    c ::F
    Δx::V
end

function steering_matrix(θ::AbstractVector, fc::Real, conf::ArrayConfig)
    (; Δx, c) = conf
    n_channels = length(Δx)
    if length(θ) == 0
        Zeros(eltype(θ), n_channels, n_channels)
    else
        Tullio.@tullio A[i,j] := exp(im*2*π*fc*sin(θ[j])*Δx[i]/c)
    end
end

function proj(θ::AbstractVector, fc::Real, conf::ArrayConfig)
    n_channels = length(conf.Δx)
    if length(θ) == 0
        Zeros(eltype(θ), n_channels, n_channels)
    else
        A = steering_matrix(θ, fc, conf)
        A*pinv(A)
    end
end

function snapshot_covariance(
    x                  ::Array,
    n_fft              ::Int,
    fs                 ::Real,
    n_temporal_snapshot::Int
)
    X_ch = map(eachcol(x)) do x_ch
        l_snap = div(length(x_ch), n_temporal_snapshot)
        stft(x_ch, l_snap, 0; nfft=n_fft)
    end
    X = cat(X_ch..., dims=3)
    @tullio Σ[i,j,n] := X[n,k,i]*conj(X[n,k,j])/size(X,2)

    Δf      = fs / n_fft
    f_range = (0:size(Σ,3)-1)*Δf

    Σ, X, f_range
end

