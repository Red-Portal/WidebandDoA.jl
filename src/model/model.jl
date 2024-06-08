
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
    Y     <: AbstractMatrix, 
    YF    <: AbstractMatrix, 
    YP    <: Real,
    Prior <: WidebandNormalGammaPrior
} <: AbstractWidebandModel
    y      ::Y
    y_fft  ::YF
    y_power::YP
    prior  ::Prior
end

struct WidebandNormalGammaParam{F <: Real}
    phi      ::F
    loglambda::F
end

function WidebandNormalGamma(y::AbstractMatrix{<:Real}, prior::WidebandNormalGammaPrior)
    Y = fft(y, 1) / sqrt(size(y, 1))
    @assert size(Y,1) == prior.n_snapshots
    P = sum(abs2, y)
    WidebandNormalGamma(y, Y, P, prior)
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
        ℓp_y = doa_diagnormal_likelihood(
            delay_filter, y_fft, y_power, ϕ, λ, alpha, beta, Δx, c,  fs
        )
        ℓp_y + ℓp_k
    elseif any((ϕ .< -π/2) .|| (ϕ .> π/2))
        -Inf
    else
        ℓp_ϕ   = k*logpdf(Uniform(-π/2, π/2), 0.0)
        ℓp_λ   = sum(Base.Fix1(logpdf, InverseGamma(alpha_lambda, beta_lambda)), λ)
        ℓjac_λ = sum(ℓλ)
        ℓp_y   = doa_diagnormal_likelihood(
            delay_filter, y_fft, y_power, ϕ, λ, alpha, beta, Δx, c, fs   
        )
        ℓp_y + ℓp_ϕ + ℓp_k + ℓp_λ + ℓjac_λ
    end
end
