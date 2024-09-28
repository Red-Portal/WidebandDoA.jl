
"""
    WidebandIsoSourcePrior(n_samples, n_fft, alpha, beta, order_prior, source_prior)

Prior for wideband signal model, where a Gaussian prior with an isotropic covariance structure is assigned on the latent source signals.

# Arguments
* `samples::Int`: Number of samples in the received signal.
* `n_fft::Int`: Length of the latent signal (\$N^{\\prime}\$ in the paper).
* `alpha::Real`: \$\\alpha\$ hyperparameter for the inverse-gamma prior on the signal variance.
* `beta::Real`: \$\\beta\$ hyperparameter for the inverse-gamma prior on the signal variance.
* `order_prior::DiscreteDistribution`: Prior on the number of sources. (Prior on \$k\$ in the paper.)
* `source_prior::UnivariateDistribution`: Hyperprior on the SNR hyperparameter of the sources. (Prior on \$\\gamma\$ in the paper.)

"""
struct WidebandIsoSourcePrior{
    F  <: Real,
    OP <: DiscreteDistribution,
    SP <: UnivariateDistribution,
} <: AbstractWidebandPrior
    n_samples   ::Int
    n_fft       ::Int
    alpha       ::F
    beta        ::F
    order_prior ::OP
    source_prior::SP
end

function logpriordensity(
    prior::WidebandIsoSourcePrior,
    θ    ::AbstractVector
)
    ϕ      = [θi.phi       for θi in θ]
    ℓλ     = [θi.loglambda for θi in θ]
    λ      = exp.(ℓλ)
    ℓjac_λ = sum(ℓλ)

    @unpack order_prior, source_prior = prior
    k    = length(ϕ)
    ℓp_k = logpdf(order_prior, k)

    if k == 0
        ℓp_k   
    elseif any((ϕ .< -π/2) .|| (ϕ .> π/2))
        -Inf
    else
        ℓp_ϕ = -k*log(π)
        ℓp_λ = sum(Base.Fix1(logpdf, source_prior), λ)
        ℓp_ϕ + ℓp_λ + ℓp_k + ℓjac_λ
    end
end

"""
    rand(rng, prior::WidebandIsoSourcePrior, )

Sample from prior with isotropic source covariance prior.

# Arguments
* `rng::Random.AbstractRNG`: Random number generator.
* `prior::WidebandIsoSourcePrior`: Prior.

# Keyword Arguments
* `k`: Model order. (Default samples from `prior.model_order`.)
* `sigma`: Signal standard deviation. (Default samples from `InverseGamma(prior.alpha, prior.beta)`.)
* `phi`: DoA of each sources. (Length must match `k`; default samples from the uniform distribution over the interval \$[-\\pi/2 , \\pi/2]\$.)
* `lambda`: SNR parameter for each source. (Length must match `k`; default samples from `prior.source_prior`.)

The sampling process is:
```math
\\begin{aligned}
    k &\\sim \\text{\\tt{}prior.model\\_order} \\\\
    \\sigma &\\sim \\text{\\textsf{Inv-Gamma}}(\\texttt{prior.alpha}, \\text{\\texttt{prior.beta}}) \\\\
    \\phi_j \\mid k &\\sim \\text{\\textsf{Uniform}}\\left(-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right) \\\\
    \\lambda_j \\mid k &\\sim \\text{\\tt{}prior.source\\_prior} \\\\
    x_j \\mid k, \\sigma, \\lambda_j &\\sim \\mathcal{N}\\left(0, \\sigma \\lambda_j \\mathrm{I} \\right)
\\end{aligned}
```

# Returns
* `params`: Parameter of wideband signal model. (keys: `k, phi, lambda, sigma, x`)

"""
function Base.rand(
    rng   ::Random.AbstractRNG,
    prior ::WidebandIsoSourcePrior;
    k     ::Int            = rand(rng, prior.order_prior),
    sigma ::Real           = rand(rng, InverseGamma(prior.alpha, prior.beta)),
    phi   ::AbstractVector = rand(rng, Uniform(-π/2, π/2), k),
    lambda::AbstractVector = rand(rng, prior.source_prior, k)
)
    @unpack n_samples, n_fft, alpha, beta, order_prior, source_prior = prior
    z_x = randn(rng, n_fft, k)
    Tullio.@tullio x[n,k] := sqrt(lambda[k])*sigma*z_x[n,k]
    (k=k, phi=phi, lambda=lambda, sigma=sigma, sourcesignals=x)
end
