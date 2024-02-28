
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
