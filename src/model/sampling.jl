
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
    N,       = n_snapshots, n_sensor
    ϕ, λ     = phi, lambda, sigma

    k   = length(ϕ)
    z_a = randn(rng, N, k)

    Tullio.@tullio a[n,k] := sqrt(λ[k])*z_a[n,k]

    simulate_propagation(rng, prior, params, a)
end

function simulate_propagation(
    rng   ::Random.AbstractRNG,
    prior ::WidebandNormalGammaPrior,
    params::NamedTuple,
    signal::AbstractMatrix,
)
    @unpack n_snapshots, order_prior, c, Δx, fs, delay_filter = prior
    @unpack k, phi, lambda, sigma = params

    N, M = n_snapshots, length(Δx)
    ϕ, σ = phi, sigma

    z_ϵ = randn(rng, N, M)

    k = length(ϕ)
    τ = inter_sensor_delay(ϕ, Δx, c)
    H = array_delay(delay_filter, τ*fs)

    A = fft(signal, 1)
    Tullio.@tullio HA[n,m] := H[n,m,k] * A[n,k]
    X = σ*HA
    x = ifft(X, 1)
    real.(x) + σ*z_ϵ
end
