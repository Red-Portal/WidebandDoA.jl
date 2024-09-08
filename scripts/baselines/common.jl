
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

function snapshot_covariance(
    x         ::Array,
    n_fft     ::Int,
    fs        ::Real,
    n_snapshot::Int
)
    l_snap = div(size(x, 1), n_snapshot)
    X_ch   = map(eachcol(x)) do x_ch
        DSP.stft(x_ch, l_snap, 0; nfft=n_fft)
    end
    # X ∈ C^{ bins × snapshot × channel }
    X = cat(X_ch..., dims=3)

    # X ∈ C^{ snapshot × channel × bins }
    X = permutedims(X, (2, 3, 1))
    @tullio Σ[i,j,n] := X[k,i,n]*conj(X[k,j,n])/size(X,1)

    Δf      = fs / n_fft
    f_range = (0:size(Σ,3)-1)*Δf

    Σ, X, f_range
end
