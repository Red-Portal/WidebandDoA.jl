
using Bootstrap
using DSP
using Distributions
using DrWatson
using FFTW
using FillArrays
using LinearAlgebra
using MKL
using Random
using ReversibleJump
using WidebandDoA

function system_setup(; use_mkl=false, is_hyper=false, start)
    if myid() > 1
       multiplier = is_hyper ? 2 : 1
       run(`taskset -pc $(multiplier*(myid() - 2) + start) $(getpid())`)
    end
    BLAS.set_num_threads(1)
end

function simulate_signal(
    rng      ::Random.AbstractRNG,
    n_samples::Int,
    n_dft    ::Int,
    ϕ        ::AbstractVector, 
    snr      ::Union{<:AbstractVector,<:Real},
    f_begin  ::Union{<:AbstractVector,<:Real},
    f_end    ::Union{<:AbstractVector,<:Real},
    fs       ::Real,
    noise_pow::Real,
    Δx       ::AbstractVector,
    c        ::Real;
    visualize::Bool = false,
)
    @assert n_dft ≥ n_samples

    n_sources = length(ϕ)
    if snr isa Real
        snr = Fill(snr, n_sources)
    end
    if f_begin isa Real
        f_begin = Fill(f_begin, n_sources)
    end
    if f_end isa Real
        f_end = Fill(f_end, n_sources)
    end
    filter = WidebandDoA.WindowedSinc(n_dft)
    x_pad  = randn(rng, n_dft, length(ϕ))
    X      = rfft(x_pad, 1)

    f_range = (0:size(X,1)-1)*fs/n_dft
    for k in 1:n_sources
        fk_begin    = view(f_begin, k)
        fk_end      = view(f_end, k)
        mask        = @. !(fk_begin ≤ f_range < fk_end)
        X[mask, k] .= zero(eltype(X))
        bw_gain     = length(mask) / (length(mask)-sum(mask))
        signal_pow  = bw_gain*10^(snr[k]/10)*noise_pow
        X[.!mask,k] *= sqrt(signal_pow)
    end
    x_pad = irfft(X, n_dft, 1)
    like  = WidebandIsoIsoLikelihood(n_dft, filter, Δx, c, fs)
    y     = rand(rng, like, x_pad, ϕ; sigma=sqrt(noise_pow))

    if visualize
        Plots.plot() |> display
        for k in 1:n_sources
            Xk = X[:,k]
            Plots.plot!((@. DSP.amp2db(max(abs(Xk/sqrt(n_dft)), 1e-10))), ylims=[-10,Inf]) |> display
        end
        signal_pow_emp = var(x_pad, dims=1)[1,:]
        @info(
            "",
            empirical_signal_power = signal_pow_emp,
            empirical_snr          = 10*log10.(signal_pow_emp/noise_pow)
        )
    end
    n_pad = div(n_dft - n_samples, 2)
    y[n_pad+1:n_pad+n_samples,:], x_pad[n_pad+1:n_pad+n_samples,:]
end

function construct_default_model(
    n_samples::Int,
    fs       ::Real;
    M        ::Int            = 20,
    spacing  ::Real           = 0.5,
    Δx       ::AbstractVector = range(0, M*spacing; length=M),
    c        ::Real           = 1500.,
    alpha    ::Real           = 0.0, 
    beta     ::Real           = 0.0, 
)
    source_prior = InverseGamma(0.01, 0.01)
    #source_prior = LogNormal(5.3, 2.3)
    order_prior = truncated(NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)), 0, M-1)
    WidebandDoA.WidebandIsoIsoModel(
        n_samples,
        Δx,
        c,
        fs,
        source_prior,
        alpha,
        beta;
        order_prior,
    )
end

function run_bootstrap(
    data′;
    stat              = mean,
    sampling_strategy = BalancedSampling(1024),
    confint_strategy  = BCaConfInt(0.95),
)
    boot = bootstrap(stat, data′, sampling_strategy)
    μ, μ_hi, μ_lo = confint(boot, confint_strategy) |> only
    (μ, μ_hi - μ, μ_lo - μ)
end

function reduce_namedtuples(f, vector_of_tuples)
    ks = keys(first(vector_of_tuples))
    @assert all(tup -> keys(tup) == ks, vector_of_tuples) 
    tuple_of_vectors = NamedTuple(k => getproperty.(vector_of_tuples, k) for k in ks)
    NamedTuple(k => f(v) for (k,v) in pairs(tuple_of_vectors))
end

function modelposterior_naive(stats)
    k_post = [stat.order for stat in stats]
    k_max  = maximum(k_post)
    sup    = 0:k_max+1
    counts = Dict{Int,Int}()

    for k in sup
        counts[k] = 0
    end

    for k in k_post
        counts[k] += 1
    end

    probs = zeros(length(sup))
    n     = length(k_post)
    for k in 0:k_max+1
        probs[k+1] = counts[k]/n
    end
    DiscreteNonParametric(sup, probs)
end
