
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
    # Λ = diagm(λ)
    #
    # -(N + ν₀)/2 log( γ₀ + y†(HΛH† + I)⁻¹y )
    # = -(N + ν₀)/2 log( γ₀ + y†(I - H(Λ⁻¹ + H†H)⁻¹H†)y )
    # = -(N + ν₀)/2 log( γ₀ + y†y - y†(H(Λ⁻¹ + H†H)⁻¹H†)y )
    #
    # det(P⊥)^{-1/2} (α/2 + y† P⊥ y)^{-MN/2 + β}
    #
    # det(P⊥) = det(H†ΛH + I) = det(Λ) det(Λ⁻¹ + H†H)
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

        Tullio.@tullio HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio Λ⁻¹pHᴴH[n,j,k] := (j == k) ? HᴴH[n,j,k] + 1/λ[k] : HᴴH[n,j,k]

        D, L = ldl_striped_matrix!(Λ⁻¹pHᴴH)

        Tullio.@tullio ℓdetΛ⁻¹pHᴴH := log(real(D[n,m]))
        if isnothing(L) || !isfinite(ℓdetΛ⁻¹pHᴴH)
            return -Inf
        end

        Tullio.@tullio ℓdetΛ := N*log(λ[i])
        ℓdetP⊥ = ℓdetΛ + ℓdetΛ⁻¹pHᴴH

        Tullio.@tullio Hᴴy[n,k]  := conj(H[n,m,k]) * Y[n,m]

        L⁻¹Hᴴy            = trsv_striped_matrix!(L, Hᴴy)
        @tullio yᴴImP⊥y := abs(L⁻¹Hᴴy[n,i]/D[n,i]*conj(L⁻¹Hᴴy[n,i]))
        yᴴP⊥y            = yᵀy - yᴴImP⊥y

        if yᴴP⊥y ≤ sqrt(eps(T))
            return -Inf
        end
        -(N*M/2 + β)*log(α/2 + yᴴP⊥y) - ℓdetP⊥/2
    end
end

