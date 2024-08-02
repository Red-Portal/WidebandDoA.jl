
using Accessors
using Distributions
using Random, Random123
using Tullio
using WidebandDoA
using ProgressMeter
using Plots

include("baselines/baselines.jl")
include("common.jl")

function test_likeratiotest()
    seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
    rng  = Random123.Philox4x(UInt64, seed, 8)
    Random123.set_counter!(rng, 1)

    ϕ = [-30, 20, 24] / 180*π

    n_fft  = 64
    n_snap = 10

    fs      = 2000
    N       = n_fft*n_snap
    M       = 15
    f0      = fs/3
    c       = 1500
    λ       = c/f0
    spacing = λ/2

    k_max  = 4
    Δx     = range(0, M*spacing; length=M)
    filter = WidebandDoA.WindowedSinc(N)
    
    ϵ   = randn(rng, N, length(ϕ))
    Δf  = f0 - (17/32*f0)
    bpf = DSP.Filters.digitalfilter(
          DSP.Filters.Bandpass(17/32*f0, f0, fs=fs), 
          DSP.Filters.Butterworth(8)
    )
    x    = mapslices(xi -> DSP.Filters.filt(bpf, xi), ϵ; dims=1)

    snrs = -8:2:8

    n_trials = 10
    l0_error = map(snrs) do snr
        prog = Progress(n_trials)
        mean(1:n_trials) do _
            σ2   = 10^(-snr/10)
            like = WidebandIsoIsoLikelihood(N, filter, Δx, c, fs)
            y    = rand(rng, like, x, ϕ; sigma=sqrt(σ2*Δf/fs))

            config        = ArrayConfig(c, Δx)
            R, Y, f_range = snapshot_covariance(y, n_fft, fs, n_snap)

            idx_sel = 12:23
            R_sel   = R[:,:,idx_sel]
            f_sel   = f_range[idx_sel]

            k, _ = likeratiotest(
                rng,
                R_sel,
                0.1,
                k_max,
                n_snap,
                f_sel,
                config;
            )
            next!(prog)
            k == length(ϕ)
        end
    end
    Plots.plot(snrs, l0_error)
end
