
# Demonstration

## System Setup

We demonstrate the use of the package.

First, let's setup the array geometry.
We will consider a system with a uniform linear array (ULA) with `M = 20` sensors and a spacing of 0.5 m, a sampling frequency of `fs = 3000`Hz, where the medium has a propagation speed of `c = 1500` m/s:

```@example demo
M  = 20
Δx = range(0, M*0.5; length=M)
c  = 1500.
fs = 3000.
nothing
```

For the target sources, we will generate `k = 4` targets, with varying bandwidths and varying SNRs:
```@example demo
k       = 4
ϕ       = [ -60,  -15,  30,  45]/180*π # True direction-of-arrivals
f_begin = [  10,  100, 500, 800]       # Source signal starting frequency
f_end   = [1000, 1500,1000, 900]       # Source signal ending frequency
snr     = [  -6,   -4,   0,   4]       # Varying SNRs in dB
nothing
```

For simulating the signals, let's use a utility function we used for the experiments.
The length of the simulated signal will be `N = 256`.
This corresponds to using `S = 8` snapshots in the paper.
```@example demo
using Random, Random123

seed = (0x97dcb950eaebcfba, 0x741d36b68bef6415)
rng  = Random123.Philox4x(UInt64, seed, 8)
Random123.set_counter!(rng, 1)

N = 256

include("../../scripts/common.jl")
y, x  = simulate_signal(
    rng, N, N*8, ϕ, snr, f_begin, f_end, fs, 1.0, Δx, c; visualize=false
)
y
```

We can visualize the simulated signal through beamforming.
Consider the Capon spectral estimator, more commonly known as minimum-variance distortionless response, or MVDR for short :
```@example demo
using Tullio
using Plots

include("../../scripts/baselines/common.jl")
include("../../scripts/baselines/subbandbeamform.jl")

N_fft  = 32
N_snap = N ÷ N_fft

R, _, f_range = snapshot_covariance(y, N_fft, fs, N_snap) # Short-time Fourier transform

config  = ArrayConfig(c, Δx)
n_grid  = 2^10
ϕ_range = range(-π/2, π/2; length=n_grid)
P       = subbandmvdr(R, ϕ_range, f_range, config)
Plots.heatmap(ϕ_range, f_range, 10*log10.(P'), xlabel="DoA (ϕ)", ylabel="Frequency")
Plots.vline!(ϕ, label="True DoAs", linecolor=:red)
savefig("angle_frequency_spectrum_plot.svg")
nothing
```
This yields the angle-frequency spectrum: 
![](angle_frequency_spectrum_plot.svg)

We can see that the signals were correctly simulated as we specified.


## Creating the Bayesian Model
Now, let's construct the Bayesian model.
We will use a non-informative inverse-gamma prior on the source SNRs ($\gamma$ in the paper), a truncated negative-binomial prior on the model order ($k$), a Jeffrey's scale prior on the signal power ($\sigma^2$) as stated in the paper:
```@example demo
alpha, beta  = 0, 0
source_prior = InverseGamma(0.01, 0.01)
order_prior  = truncated(NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)), 0, M-1)
model        = WidebandDoA.WidebandIsoIsoModel(
    N, Δx, c, fs, source_prior, alpha, beta; order_prior
)
nothing
```

The model can be conditioned on the data as follows:
```@example demo
cond  = WidebandConditioned(model, y)
nothing
```

We are now ready to infer the posterior for this model.

## Inference with RJMCMC
For inference, we will use the [`ReversibleJump`](https://github.com/Red-Portal/ReversibleJump.jl) package, which is the inference counterpart of this package.

We use independent jump proposals with the uniform-log-normal auxiliary proposal distributions ($q(\gamma)$, $q(\gamma)$ in the paper) as done in the paper:
```@example demo
using ReversibleJump

prop = UniformNormalLocalProposal(0.0, 2.0)
jump = IndepJumpProposal(prop)
nothing
```
For the *update move*, we will use slice sampling[^N2003] with the stepping-out procedure:
```@example demo
mcmc = SliceSteppingOut([2.0, 2.0])
nothing
```

[^N2003]: Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.

The RJMCMC algorithm we use is the non-reversible jump algorithm by Gagnon and Doucet[^GD2020].
```@example demo
rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
nothing
```
[^GD2020]: Gagnon, Philippe, and Arnaud Doucet. "Nonreversible jump algorithms for Bayesian nested model selection." Journal of Computational and Graphical Statistics 30.2 (2020): 312-323.

Now let's simulate some Markov chains!
```@example demo
n_samples = 4_000

initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
initial_order  = 0
samples, stats = ReversibleJump.sample(
    rng,
    rjmcmc,
    cond,
    n_samples,
    initial_order,
    initial_params;
    show_progress=false,
)
samples
```

## Signal Detection
Detection can be performed by analyzing the posterior of the model order ($k$).

Here is the trace of the model order:
```@example demo
Plots.plot([stat.order for stat in stats],  xlabel="RJMCMC Iteration",  ylabel="Model order")
savefig("model_order_trace.svg")
nothing
```
![](model_order_trace.svg)

And the histogram:
```@example demo
Plots.histogram([stat.order for stat in stats], xlabel="order", normed=true)
savefig("model_order_hist.svg")
nothing
```
![](model_order_hist.svg)

From the posterior, we can obtain point estimates of the model order $k$.
In the paper, we use the median.

## DoA Estimation
Now, let's estimate the DoAs.

For this, we will discard the first 10% of the samples and only use the remaining samples.
```@example demo
burned = samples[n_samples ÷ 10:end]
```

*Bayesian model averaging* (BMA) correponds to flattening all the local variables:
```@example demo
flat = vcat(burned...)
```
On the other hand, *Bayesian model selection* corresponds to selecting the samples that have a specific model order.
	
Here is the marginal posterior of the DoAs:
```@example demo
Plots.histogram([θ.phi for θ in flat], normed=true, bins=256, xlims=[-π/2, π/2], xlabel="DoA (ϕ)", label="Posterior", ylabel="Density")
Plots.vline!(ϕ, label="True", color=:red, linestyle=:dash)
savefig("doa_hist.svg")
nothing
```
![](doa_hist.svg)

Unfortunately, this does not yield the DoA posterior corresponding to each *source*.
For this, we turn to the relabeling algorithm by Roodaki *et al.*[^RBF2014].
This algorithm fits a Gaussian mixture model on the histogram above.
It also generates labels for each local variable, so that we can label variable other than just the DoAs.
For this though, we have to choose the number of mixture components. 
Roodaki *et al.* recommend the 80% or 90% upper percentile of the posterior:
```@example demo
k_mixture = quantile([stat.order for stat in stats], 0.9) |> Base.Fix1(round, Int)
ϕ_post    = [[target.phi for target in sample] for sample in burned]
mixture, labels = WidebandDoA.relabel(
    rng, ϕ_post, k_mixture; show_progress=false
)
mixture
```
[^RBF2014]: Roodaki, Alireza, Julien Bect, and Gilles Fleury. "Relabeling and summarizing posterior distributions in signal decomposition problems when the number of components is unknown." *IEEE Transactions on Signal Processing* (2014).

Let's check that the Gaussian mixture approximation is accurate.
This can be done by comparing the marginal density of the mixture against the histogram:
```@example demo
using StatsPlots

Plots.stephist([θ.phi for θ in flat], normed=true, bins=256, xlims=[-π/2, π/2], xlabel="DoA (ϕ)", ylabel="Density", label="Posterior", fill=true)
Plots.plot!(range(-π/2,π/2; length=1024), Base.Fix1(pdf, mixture), label="Mixture Approximation", linewidth=2)
savefig("doa_relabel_density.svg")
nothing
```
![](doa_relabel_density.svg)

The DoA posterior of each source is then represented by each component of the mixture:
```@example demo
Plots.plot(mixture, label="Component", fill=true)
Plots.vline!(ϕ, label="True", color=:red, linestyle=:dash)
savefig("doa_relabel_comp.svg")
nothing
```
![](doa_relabel_comp.svg)


## Reconstruction
Finally, we will demonstrate reconstruction.
Unfortunately, the API for reconstruction is a little less ironed-out, but it is functional.

For this, we have to use the labels generated by the relabeling procedure.
Here, for each RJMCMC sample, we will sample from the conditional posterior conditional and then relabel the signal segments to their corresponding source.
We also thin the samples by a factor `n_thin = 100` to speed up things and reduce memory consumption.
```@example demo
x_samples = [Vector{Float64}[] for j in 1:k_mixture]
x_means   = [Vector{Float64}[] for j in 1:k_mixture]

n_thin = n_samples ÷ 10

samples_thinned = burned[1:n_thin:end]
labels_thinned  = labels[1:n_thin:end]

for (sample, labs) in zip(samples_thinned, labels_thinned)
    dist_x   = WidebandDoA.reconstruct(cond, sample)
    x_sample = rand(rng, dist_x) # Conditional posterior sample
    x_mean   = mean(dist_x)      # Conditional posterior mean

    kj        = length(sample)
    total_len = length(x_sample)
    blocksize = total_len ÷ kj

    # Labeling the conditional posterior mean and sample
    for (idx, label) in enumerate(labs)
        if label > k_mixture
            # A label of k_mixture + 1 corresponds to clutter
            continue
        end

        # The source signals are flattened so we have to slice the block corresponding
        # to the source the label is pointing to.
        blockrange = (idx-1)*blocksize+1:idx*blocksize
        xj_sample  = x_sample[blockrange[1:N]]
        xj_mean    = x_mean[  blockrange[1:N]]
        push!(x_samples[label], xj_sample)
        push!(x_means[label],   xj_mean)
    end
end
nothing
```

Now that sampling and relabeling is done, we can visualize the posterior samples against with the minimum mean-squared error (MMSE) estimates (posterior mean).
The variability of the posterior samples represent the posterior uncertainty:
```@example demo
x_mmse = mean.(x_samples)

plts = map(1:k_mixture) do j
    p = Plots.plot(x_mmse[j], linecolor=:blue, label="MMSE", xlabel="Sample index")
    for x_sample in x_samples[j]
        Plots.plot!(p, x_sample, linecolor=:blue, alpha=0.5, linewidth=0.2, label=nothing)
    end
    p
end
Plots.plot(plts..., layout = (4, 1))
savefig("recon_samples.svg")
nothing
```
![](recon_samples.svg)

Finally, let's compare the MMSE against the ground truth `x`.
```@example demo
Plots.plot( x_mmse, layout=(4,1), label="MMSE", xlabel="Sample Index")
Plots.plot!(x,      layout=(4,1), label="True", xlabel="Sample Index")
savefig("recon_mmse_comparison.svg")
nothing
```
![](recon_mmse_comparison.svg)
