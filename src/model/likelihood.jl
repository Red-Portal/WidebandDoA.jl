
function doa_normalgamma_likelihood(
    filter ::AbstractDelayFilter,
    Y      ::AbstractMatrix{Complex{T}}, 
    yᵀy    ::T,
    ϕ      ::AbstractVector{T},
    λ      ::AbstractVector{T},
    α      ::Real,
    β      ::Real,
    Δx     ::AbstractVector,
    c, 
    fs
) where {T <: Real}
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
        -(N*M/2 + α)*log(β/2 + yᵀy)
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

function ReversibleJump.logdensity(
    model::WidebandNormalGamma,
    θ    ::AbstractVector{<:WidebandNormalGammaParam}
)
    @unpack y_fft, y_power, prior = model
    @unpack delay_filter, Δx, c, fs, alpha, beta, alpha_lambda, beta_lambda, order_prior = prior

    k  = length(θ)
    ϕ  = [θj.phi       for θj in θ]
    ℓλ = [θj.loglambda for θj in θ]
    λ  = exp.(ℓλ)

    ℓp_k = logpdf(order_prior, k)

    if ℓp_k == -Inf
        -Inf
    elseif k == 0
        ℓp_y = doa_normalgamma_likelihood(
            delay_filter, y_fft, y_power, ϕ, λ, alpha, beta, Δx, c,  fs   
        )
        ℓp_y + ℓp_k
    elseif any((ϕ .< -π/2) .|| (ϕ .> π/2))
        -Inf
    else
        λ      = exp.(ℓλ)
        ℓp_ϕ   = k*logpdf(Uniform(-π/2, π/2), 0.0)
        ℓp_λ   = sum(Base.Fix1(logpdf, Gamma(alpha_lambda, beta_lambda)), λ)
        ℓjac_λ = sum(ℓλ)
        ℓp_y   =  doa_normalgamma_likelihood(
            delay_filter, y_fft, y_power, ϕ, λ, alpha, beta, Δx, c,  fs   
        )
        ℓp_y + ℓp_ϕ + ℓp_λ + ℓjac_λ + ℓp_k
    end
end
