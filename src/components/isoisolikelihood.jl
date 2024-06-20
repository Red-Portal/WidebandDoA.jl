
struct WidebandIsoIsoLikelihood{
    DF <: AbstractDelayFilter,
    AO <: AbstractVector,
    F  <: Real
} <: AbstractWidebandLikelihood
    n_snapshots ::Int
    delay_filter::DF
    Δx          ::AO
    c           ::F
    fs          ::F
end

function loglikelihood(
    likelihood::WidebandIsoIsoLikelihood,
    prior     ::WidebandIsoSourcePrior,
    data      ::WidebandData,
    params,
)
    @unpack y_fft, y_power           = data
    @unpack delay_filter, Δx, c,  fs = likelihood
    @unpack alpha, beta,             = prior

    ϕ = [param.phi            for param in params]
    λ = [exp(param.loglambda) for param in params]

    # Isotropic Gaussian noise with isotropic Gaussian source prior
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

    N = size(y_fft,1)
    M = size(y_fft,2)
    K = length(ϕ)

    if K == 0
        -(N*M/2 + beta)*log(alpha/2 + y_power)
    else
        τ  = inter_sensor_delay(ϕ, Δx, c)
        Δn = τ*fs
        H  = array_delay(delay_filter, Δn)

        Tullio.@tullio HᴴH[n,j,k]     := conj(H[n,m,j]) * H[n,m,k]
        Tullio.@tullio Λ⁻¹pHᴴH[n,j,k] := (j == k) ? HᴴH[n,j,k] + 1/λ[k] : HᴴH[n,j,k]

        D, L = ldl_striped_matrix!(Λ⁻¹pHᴴH)

        Tullio.@tullio ℓdetΛ⁻¹pHᴴH := log(real(D[n,m]))
        if isnothing(L) || !isfinite(ℓdetΛ⁻¹pHᴴH)
            return -Inf
        end

        Tullio.@tullio ℓdetΛ := N*log(λ[i])
        ℓdetP⊥ = ℓdetΛ + ℓdetΛ⁻¹pHᴴH

        Tullio.@tullio Hᴴy[n,k]  := conj(H[n,m,k]) * y_fft[n,m]

        L⁻¹Hᴴy            = trsv_striped_matrix!(L, Hᴴy)
        @tullio yᴴImP⊥y := abs(L⁻¹Hᴴy[n,i]/D[n,i]*conj(L⁻¹Hᴴy[n,i]))
        yᴴP⊥y            = y_power - yᴴImP⊥y

        if yᴴP⊥y ≤ eps(eltype(data.y))
            return -Inf
        end
        -(N*M/2 + beta)*log(alpha/2 + yᴴP⊥y) - ℓdetP⊥/2
    end
end


"""
    rand(rng, likelihood, params)

The sampling process is as follows:
```math
\\begin{aligned}
    a         &\\sim \\mathcal{N}(0, \\sigma^2 \\Lambda) \\\\
    \\epsilon &\\sim \\mathcal{N}(0, \\sigma^2 \\mathrm{I}) \\\\
    x         &= H a \\\\
    y         &= x + \\epsilon  
\\end{aligned}
```

After marginalizing out the source signal magnitudes \$a\$, 
```math
\\begin{aligned}
    \\epsilon &\\sim \\mathcal{N}(0, \\sigma^2 \\mathrm{I}) \\\\
    x         &\\sim \\mathcal{N}(0, \\sigma^2 H \\Lambda H^{\\top}) \\\\
    y         &= x + \\epsilon  
\\end{aligned}
```
and the noise \$\\epsilon\$,
```math
\\begin{aligned}
    y         &\\sim \\mathcal{N}(0, \\sigma^2 \\left( H \\Lambda H^{\\top} + \\mathrm{I} \\right)).
\\end{aligned}
```
Sampling from this distribution is as simple as
```math
\\begin{aligned}
  y = \\sigma H \\Lambda^{1/2} z_x + \\sigma z_{\\epsilon},
\\end{aligned}
```
where \$z_x\$ and \$z_{\\epsilon}\$ are independent standard Gaussian vectors.
"""
function Base.rand(
    rng       ::Random.AbstractRNG,
    likelihood::WidebandIsoIsoLikelihood,
    prior     ::WidebandIsoSourcePrior,
    x         ::AbstractMatrix,
    phi       ::AbstractVector;
    sigma     ::Real = rand(rng, InverseGamma(prior.alpha, prior.beta)),
)
    @unpack n_snapshots, delay_filter, Δx, c, fs = likelihood

    N, M = n_snapshots, length(Δx)
    ϕ, σ = phi, sigma

    z_ϵ = randn(rng, N, M)

    k = length(ϕ)
    τ = inter_sensor_delay(ϕ, Δx, c)
    H = array_delay(delay_filter, τ*fs)

    X  = fft(x, 1)
    Tullio.@tullio HX[n,m] := H[n,m,k] * X[n,k]
    Hx = ifft(HX, 1)
    real.(Hx) + σ*z_ϵ
end
