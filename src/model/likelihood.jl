
function doa_diagnormal_likelihood(
    filter::AbstractDelayFilter,
    Y     ::AbstractMatrix{Complex{T}}, 
    yᵀy   ::T,
    ϕ     ::AbstractVector{T},
    λ     ::AbstractVector{T},
    α     ::Real,
    β     ::Real,
    Δx    ::AbstractVector,
    c, 
    fs
) where {T <: Real}
    # Normal with Diagonal Covariance prior
    #
    # W = diagm(λ)
    #
    # -(N + ν₀)/2 log( γ₀ + y†(H†WH + I)y )
    # = -(N + ν₀)/2 log( γ₀ + y†(I - H(W⁻¹ + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H(W⁻¹ + H†H)⁻¹H†)y )
    #
    # det(P⊥)^{-1/2} (α/2 + y† P⊥ y)^{-MN/2 + β}
    #
    # det(P⊥) = det(H†WH + I) = det(W) det(W⁻¹ + H†H)
    #

    N = size(Y,1)
    M = size(Y,2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + β)*log(α/2 + yᵀy)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = τ*fs
        H  = array_delay(filter, Δn)

        Tullio.@tullio threads=false HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio threads=false W⁻¹pHᴴH[n,j,k] := (j == k) ? HᴴH[n,j,k] + 1/λ[k] : HᴴH[n,j,k]

        W⁻¹pHᴴH⁻¹ = W⁻¹pHᴴH
        W⁻¹pHᴴH⁻¹, ℓdetW⁻¹pHᴴH = inv_hermitian_striped_matrix!(W⁻¹pHᴴH⁻¹)
        if isnothing(W⁻¹pHᴴH⁻¹) || !isfinite(ℓdetW⁻¹pHᴴH)
            return -Inf
        end

        Tullio.@tullio threads=false ℓdetW := N*log(λ[i])
        ℓdetP⊥ = ℓdetW + ℓdetW⁻¹pHᴴH

        Tullio.@tullio threads=false Hᴴy[n,k] := conj(H[n,m,k]) * Y[n,m]
        Tullio.@tullio threads=false Py[n,k]  := W⁻¹pHᴴH⁻¹[n,k,j] * Hᴴy[n,j]

        Tullio.@tullio threads=false yᴴP⊥y := real(conj(Hᴴy[i])*Py[i])
        yᴴImP⊥y = yᵀy - yᴴP⊥y

        if yᴴImP⊥y ≤ 0 
            return -Inf
        end

        -(N*M/2 + β)*log(α/2 + yᴴImP⊥y) - ℓdetP⊥/2
    end
end

