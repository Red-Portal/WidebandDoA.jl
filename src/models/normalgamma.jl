
struct WidebandNormalGammaPrior{
    N  <: Integer,
    DF <: AbstractDelayFilter,
    AO <: AbstractVector,
    SS,
    FS,
    OP <: DiscreteDistribution,
    A  <: Real,
    B  <: Real,
    LA <: Real,
    LB <: Real,
}
    n_snapshots ::N
    delay_filter::DF
    Δx          ::AO
    c           ::SS
    fs          ::FS
    order_prior ::OP
    alpha       ::A
    beta        ::B
    alpha_lambda::LA
    beta_lambda ::LB
end

struct WidebandNormalGamma{
    YF    <: AbstractMatrix, 
    YP    <: Real,
    Prior <: WidebandNormalGammaPrior
} <: AbstractWidebandModel
    y_fft  ::YF
    y_power::YP
    prior  ::Prior
end

function sample_params(rng::Random.AbstractRNG, prior::WidebandNormalGammaPrior)
    @unpack order_prior, alpha, beta, alpha_lambda, beta_lambda = prior
    k = rand(rng, order_prior)
    ϕ = rand(rng, Uniform(-π/2, π/2), k)
    λ = rand(rng, InverseGamma(alpha_lambda, beta_lambda), k)
    σ = rand(rng, InverseGamma(alpha, beta))
    (k=k, phi=ϕ, lambda=λ, sigma=σ)
end

"""
    sample_signal(rng, prior, params)

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
  y = \\sigma H \\Lambda^{1/2} z_a + \\sigma z_{\\epsilon},
\\end{aligned}
```
where \$z_a\$ and \$z_{\\epsilon}\$ are independent standard Gaussian vectors.
"""
function sample_signal(
    rng   ::Random.AbstractRNG,
    prior ::WidebandNormalGammaPrior,
    params::NamedTuple,
)
    @unpack n_snapshots, order_prior, c, Δx, fs, delay_filter = prior
    @unpack k, phi, lambda, sigma = params

    n_sensor = length(Δx)
    N, M     = n_snapshots, n_sensor
    ϕ, λ, σ  = phi, lambda, sigma

    k = length(ϕ)
    τ = inter_sensor_delay(ϕ, Δx, c)
    H = array_delay(delay_filter, τ*fs)

    z_a = randn(rng, N, k)
    z_ϵ = randn(rng, N, M)

    Tullio.@tullio a[n,k] := λ[k]*z_a[n,k]
    A = fft(a, 1)

    Tullio.@tullio HA[n,m] := H[n,m,k] * A[n,k]
    X = σ*HA
    x = ifft(X, 1)
    y = real.(x) + σ*z_ϵ
    y
end

function WidebandNormalGamma(y::AbstractMatrix{<:Real}, prior::WidebandNormalGammaPrior)
    Y = fft(y, 1) / sqrt(size(y, 1))
    @assert size(Y,1) == prior.n_snapshots
    P = sum(abs2, y)
    WidebandNormalGamma(Y, P, prior)
end

function WidebandNormalGamma(
    y    ::AbstractMatrix,
    Δx,
    c,
    fs,
    α_λ  ::Real,
    β_λ  ::Real,
    α    ::Real = 0,
    β    ::Real = 0;
    delay_filter::AbstractDelayFilter  = WindowedSinc(size(y,1)),
    order_prior ::DiscreteDistribution = NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1))
)
    WidebandNormalGamma(
        y, WidebandNormalGammaPrior(
            size(y, 1), delay_filter, Δx, c, fs, order_prior, α, β, α_λ, β_λ
        )
    )
end

struct WidebandNormalGammaParam{F <: Real}
    phi      ::F
    loglambda::F
end

struct UniformNormalLocalProposal{F <: Real}
    mu   ::F
    sigma::F
end

function ReversibleJump.local_proposal_sample(
    rng  ::Random.AbstractRNG,
         ::WidebandNormalGamma,
    prop ::UniformNormalLocalProposal
)
    WidebandNormalGammaParam(
        rand(rng, Uniform(-π/2, π/2)),
        rand(rng, Normal(prop.mu, prop.sigma))
    )
end

function ReversibleJump.local_proposal_logpdf(
         ::WidebandNormalGamma,
    prop ::UniformNormalLocalProposal,
    θ    ::AbstractVector{<:WidebandNormalGammaParam},
    j    ::Integer
)
    ℓλ = θ[j].loglambda
    -log(π) + logpdf(Normal(prop.mu, prop.sigma), ℓλ)
end

function ReversibleJump.local_insert(
      ::WidebandNormalGamma,
    θ ::AbstractVector{<:WidebandNormalGammaParam},
    j ::Integer,
    θj::WidebandNormalGammaParam)
    insert!(copy(θ), j, θj)
end

function ReversibleJump.local_deleteat(
     ::WidebandNormalGamma,
    θ::AbstractVector{<:WidebandNormalGammaParam},
    j::Integer
)
    deleteat!(copy(θ), j), θ[j]
end

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

struct WidebandNormalGammaConstantOrder{M}
    model::M
    order::Int
end

function logdensity(
    wrapper::WidebandNormalGammaConstantOrder,
    θ      ::AbstractVector{T}
) where {T <: Real}
    model  = wrapper.model
    order  = length(θ) ÷ 2
    params = [WidebandNormalGammaParam{T}(θ[i], θ[order+i]) for i in 1:order]
    ReversibleJump.logdensity(model, params)
end

function ReversibleJump.transition_mcmc(
    rng    ::Random.AbstractRNG,
    sampler::AbstractSliceSampling,
    model, θ::AbstractVector{WidebandNormalGammaParam{T}}
) where {T<:Real}
    window_base     = sampler.window
    order           = length(θ)
    window          = vcat(
        fill(first(window_base), order),
        fill(last(window_base), order)
    )
    sampler_adapted = @set sampler.window = window
    model_wrapper   = WidebandNormalGammaConstantOrder(model, order)
    ϕ               = T[θj.phi       for θj in θ]
    ℓλ              = T[θj.loglambda for θj in θ]
    θ_flat          = vcat(ϕ, ℓλ)
    θ_flat, ℓp, _   = slice_sampling(rng, sampler_adapted, model_wrapper, θ_flat)
    θ               = WidebandNormalGammaParam{T}[
        WidebandNormalGammaParam(θ_flat[i], θ_flat[order+i]) for i in 1:order
    ]
    θ, ℓp
end
