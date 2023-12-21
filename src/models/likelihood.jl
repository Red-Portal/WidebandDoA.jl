
function inter_sensor_delay(ϕ::AbstractVector, 
                            Δx::AbstractVector, 
                            c)
    Tullio.@tullio avx=64 threads=false τ[m,k] := Δx[m]*sin(ϕ[k])/c
end

function delay_filters(Δn::Matrix{T}, n_fft::Int)::Array{Complex{T},3} where {T <: Real}
#=
    Returns the fourier domain fractional delay filters as a matrix
        H ∈ R^{ N × M × K }
    The fractional delay filters are the ones in:

    S. -C. Pei and Y. -C. Lai, "Closed Form Variable Fractional Time Delay Using FFT," 
    in IEEE Signal Processing Letters, 2012.
    
    S. -C. Pei and Y. -C. Lai, 
    "Closed form variable fractional delay using FFT with transition band trade-off," 
    IEEE International Symposium on Circuits and Systems (ISCAS), 2014.

    Note: Technically speaking this filter is now overkill.
=##
    θ    = collect(0:n_fft-1)*2*π/n_fft
    a_fd = T(0.25)
    Tullio.@tullio avx=64 threads=false H[n,m,k] := begin
        if (n - 1) <= floor(Int, n_fft/2)
            if (n - 1) == 0
                Complex{T}(1.0)
            elseif (n - 1) <= ceil(Int, n_fft/2) - 2
                exp(-1im*Δn[m,k]*θ[n])
            elseif (n - 1) <= ceil(Int, n_fft/2) - 1
                a_fd*cos(Δn[m,k]*π) + (1 - a_fd)*exp(-1im*Δn[m,k]*2*π/n_fft*(n_fft/2 - 1))
            elseif (n - 1) == ceil(Int, n_fft/2)
                Complex{T}(cos(Δn[m,k]*π))
            end
        else
            zero(Complex{T})
        end
    end
    idx_begin_cplx = ceil(Int, n_fft/2) + 1
    H[idx_begin_cplx:end,:,:] = begin
        if isodd(n_fft)
            conj.(H[idx_begin_cplx-1:-1:2,:,:])
        else
            conj.(H[idx_begin_cplx:-1:2,:,:])
        end
    end
    H
    # ω = collect(0:n_fft-1)*2*π/n_fft
    # Tullio.@tullio H[n,m,k] := exp(-1im*Δn[m,k]*ω[n])
end

function doa_blockg_likelihood(Y::AbstractMatrix{Complex{T}}, 
                               yᵀy::T,
                               ϕ::AbstractVector{T},
                               g::AbstractVector{T},
                               α::Real,
                               β::Real,
                               Δx::AbstractVector,
                               c, 
                               fs) where {T <: Real}
    # Block g-prior
    #
    # G = diag( g₁(H₁†H₁)⁻¹ ... gₖ(Hₖ†Hₖ)⁻¹ )
    #
    # -(N + ν₀)/2 log( γ₀ + y†(H†GH + I)H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†(I - H(G⁻¹ + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H(G⁻¹ + H†H)⁻¹H†)y )

    N = size(Y,1)
    M = size(Y,2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + β)*log(α/2 + yᵀy)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = NoUnits.(τ*fs) |> Matrix{T}
        H  = delay_filters(Δn, N)

        Tullio.@tullio threads=false G⁻¹[n,k]       := 1/g[k]*real(conj(H[n,m,k])*H[n,m,k])
        Tullio.@tullio threads=false HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio threads=false G⁻¹pHᴴH[n,j,k] := (j == k) ? HᴴH[n,j,k] + G⁻¹[n,k] : HᴴH[n,j,k]

        G⁻¹pHᴴH⁻¹ = G⁻¹pHᴴH
        G⁻¹pHᴴH⁻¹, ℓdetG⁻¹pHᴴH = inv_hermitian_striped_matrix!(G⁻¹pHᴴH⁻¹)
        if isnothing(G⁻¹pHᴴH⁻¹) || !isfinite(ℓdetG⁻¹pHᴴH)
            return -Inf
        end

        Tullio.@tullio threads=false ℓdetG⁻¹ := log(abs(G⁻¹[i]))
        ℓdetfactor = ℓdetG⁻¹ - ℓdetG⁻¹pHᴴH

        Tullio.@tullio threads=false Hᴴy[n,k] := conj(H[n,m,k]) * Y[n,m]
        Tullio.@tullio threads=false Py[n,k]  := G⁻¹pHᴴH⁻¹[n,k,j] * Hᴴy[n,j]

        Tullio.@tullio yᴴP⊥y := real(conj(Hᴴy[i])*Py[i])
        yᴴImP⊥y = yᵀy - yᴴP⊥y

        if yᴴImP⊥y ≤ 0 
            return -Inf
        end

        -(N*M/2 + α)*log(β/2 + yᴴImP⊥y) + ℓdetfactor/2
    end
end

function doa_normalgamma_likelihood(Y::AbstractMatrix{Complex{T}}, 
                                    yᵀy::T,
                                    ϕ::AbstractVector{T},
                                    λ::AbstractVector{T},
                                    α::Real,
                                    β::Real,
                                    Δx::AbstractVector,
                                    c, 
                                    fs) where {T <: Real}
    # Normal-gamma prior
    #
    # W = diagm(λ)
    #
    # -(N + ν₀)/2 log( γ₀ + y†(H†WH + I)y )
    # = -(N + ν₀)/2 log( γ₀ + y†(I - H(W⁻¹ + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H(W⁻¹ + H†H)⁻¹H†)y )
    #
    # det((H†WH + I)⁻¹) = det(W⁻¹) / det(W⁻¹ + H†H)

    N = size(Y,1)
    M = size(Y,2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + β)*log(α/2 + yᵀy)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = NoUnits.(τ*fs) |> Matrix{T}
        H  = delay_filters(Δn, N)

        Tullio.@tullio threads=false HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio threads=false W⁻¹pHᴴH[n,j,k] := (j == k) ? HᴴH[n,j,k] + 1/λ[k] : HᴴH[n,j,k]

        W⁻¹pHᴴH⁻¹ = W⁻¹pHᴴH
        W⁻¹pHᴴH⁻¹, ℓdetW⁻¹pHᴴH = inv_hermitian_striped_matrix!(W⁻¹pHᴴH⁻¹)
        if isnothing(W⁻¹pHᴴH⁻¹) || !isfinite(ℓdetW⁻¹pHᴴH)
            return -Inf
        end

        Tullio.@tullio threads=false ℓdetW⁻¹ := -N*log(λ[i])
        ℓdetfactor = ℓdetW⁻¹ - ℓdetW⁻¹pHᴴH

        Tullio.@tullio threads=false Hᴴy[n,k] := conj(H[n,m,k]) * Y[n,m]
        Tullio.@tullio threads=false Py[n,k]  := W⁻¹pHᴴH⁻¹[n,k,j] * Hᴴy[n,j]

        Tullio.@tullio threads=false yᴴP⊥y := real(conj(Hᴴy[i])*Py[i])
        yᴴImP⊥y = yᵀy - yᴴP⊥y

        if yᴴImP⊥y ≤ 0 
            return -Inf
        end

        -(N*M/2 + α)*log(β/2 + yᴴImP⊥y) + ℓdetfactor/2
    end
end

function doa_ridgegprior_likelihood(Y::AbstractMatrix{Complex{T}}, 
                                    yᵀy::T,
                                    ϕ::AbstractVector{T},
                                    g::T,
                                    δ::Real,
                                    α::Real,
                                    β::Real,
                                    Δx::AbstractVector,
                                    c, 
                                    fs) where {T <: Real}
    # Ridge g-prior
    #
    # G   = g (HᴴH + λI)⁻¹
    # G⁻¹ = 1/g (HᴴH + λI)
    #
    # -(N + ν₀)/2 log( γ₀ + y†(H†GH + I)y )
    # = -(N + ν₀)/2 log( γ₀ + y†(I - H(G⁻¹ + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H(g⁻¹ HᴴH + λ/g I + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H((1 + g⁻¹) HᴴH + λ/g I)⁻¹H†)y )
    #
    # det((HᴴGH + I)⁻¹) = det(G⁻¹) / det(G⁻¹ + H†H)

    N = size(Y,1)
    M = size(Y,2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + β)*log(α/2 + yᵀy)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = NoUnits.(τ*fs) |> Matrix{T}
        H  = delay_filters(Δn, N)

        Tullio.@tullio threads=false HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio threads=false G⁻¹pHᴴH[n,j,k] := (j == k) ? (1 + 1/g)*HᴴH[n,j,k] + δ/g : (1 + 1/g)*HᴴH[n,j,k]
        Tullio.@tullio threads=false G⁻¹[n,j,k]     := (j == k) ? 1/g*HᴴH[n,j,k] + δ/g : 1/g*HᴴH[n,j,k]

        G⁻¹pHᴴH⁻¹ = G⁻¹pHᴴH
        G⁻¹pHᴴH⁻¹, ℓdetG⁻¹pHᴴH = inv_hermitian_striped_matrix!(G⁻¹pHᴴH⁻¹)
        if isnothing(G⁻¹pHᴴH⁻¹) || !isfinite(ℓdetG⁻¹pHᴴH)
            return -Inf
        end

        G, ℓdetG⁻¹ = inv_hermitian_striped_matrix!(G⁻¹)
        if isnothing(G) || !isfinite(ℓdetG⁻¹)
            return -Inf
        end
        ℓdetfactor = ℓdetG⁻¹ - ℓdetG⁻¹pHᴴH

        Tullio.@tullio threads=false Hᴴy[n,k] := conj(H[n,m,k]) * Y[n,m]
        Tullio.@tullio threads=false Py[n,k]  := G⁻¹pHᴴH⁻¹[n,k,j] * Hᴴy[n,j]

        Tullio.@tullio threads=false yᴴP⊥y = real(conj(Hᴴy[i])*Py[i])
        yᴴImP⊥y = yᵀy - yᴴP⊥y

        if yᴴImP⊥y ≤ 0 
            return -Inf
        end

        -(N*M/2 + α)*log(β/2 + yᴴImP⊥y) + ℓdetfactor/2
    end
end
