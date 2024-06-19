
using Bootstrap
using Distributions
using DrWatson
using DSP
using LinearAlgebra
using MKL
using ReversibleJump
using WidebandDoA

function system_setup(; use_mkl=false, is_hyper=false, start)
    if myid() > 1
       multiplier = is_hyper ? 2 : 1
       run(`taskset -pc $(multiplier*(myid() - 2) + start) $(getpid())`)
    end
    BLAS.set_num_threads(1)
end

function construct_default_model(
    rng::Random.AbstractRNG, ϕ::AbstractVector, snr::Real
)
    N      = 32
    M      = 20
    Δx     = range(0, M*0.5; length=M)
    c      = 1500
    fs     = 1000
    
    filter = WidebandDoA.WindowedSinc(N)
    λ      = fill(10^(snr/10), length(ϕ))
    σ      = 1.0

    # P[λ > 0.1] = 99%
    α_λ, β_λ = 2.1, 0.6823408279481948
    α, β     = 0., 0.0

    order_prior = truncated(NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)), 0, M-1)
    model       = WidebandDoA.WidebandNormalGammaPrior(
        N, filter, Δx, c, fs, order_prior, α_λ, β_λ, α, β,
    )

    θ = (k=length(ϕ), phi=ϕ, lambda=λ, sigma=σ)
    y = WidebandDoA.sample_signal(rng, model, θ)

    model = WidebandDoA.WidebandNormalGamma(
        y, Δx, c, fs, α_λ, β_λ, α, β;
        delay_filter=filter,
        order_prior =order_prior
    )
    model, θ
end

function run_bootstrap(
    data′;
    sampling_strategy = BalancedSampling(1024),
    confint_strategy  = PercentileConfInt(0.8),
)
    boot = bootstrap(mean, data′, sampling_strategy)
    μ, μ_hi, μ_lo = confint(boot, confint_strategy) |> only
    (μ, μ_hi - μ, μ_lo - μ)
end

function reduce_namedtuples(f, vector_of_tuples)
    ks = keys(first(vector_of_tuples))
    @assert all(tup -> keys(tup) == ks, vector_of_tuples) 
    tuple_of_vectors = NamedTuple(k => getproperty.(vector_of_tuples, k) for k in ks)
    NamedTuple(k => f(v) for (k,v) in pairs(tuple_of_vectors))
end

function sample_bandlimited_signals(
    rng    ::Random.AbstractRNG,
    prior  ::WidebandDoA.WidebandNormalGammaPrior,
    params ::NamedTuple,
    f_begin::Real,
    f_end  ::Real,
)
    @unpack n_snapshots, order_prior, c, Δx, fs, delay_filter = prior
    @unpack k, phi, lambda, sigma = params
    
    N        = n_snapshots
    ϕ, λ, σ  = phi, lambda, sigma
    k        = length(ϕ)

    bpf = DSP.Filters.digitalfilter(
        DSP.Filters.Bandpass(f_begin, f_end, fs=fs), 
        DSP.Filters.Butterworth(8)
    )
    z_a  = randn(rng, 4*N, k)
    gain = sqrt((fs/2)/(f_end - f_begin))
    a    = mapreduce(hcat, zip(λ, eachcol(z_a))) do (λj, z_aj)
        aj = gain*sqrt(λj)*z_aj
        reshape(DSP.Filters.filt(bpf, aj)[end-N+1:end], (:,1))
    end
    y = WidebandDoA.simulate_propagation(rng, prior, params, a)
    y, a
end
