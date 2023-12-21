
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
