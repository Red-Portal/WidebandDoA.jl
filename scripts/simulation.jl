### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 71585db6-d5de-11ee-2f91-6306270cfdbb
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ fd1c1453-8293-4371-9296-8b3b8baad279
begin
	using Accessors
	using Plots, StatsPlots
	using Distributions
	using Random
	using MKL
    using Tullio
	# using LoopVectorizations
	using ReversibleJump
	using WidebandDoA
	using StableRNGs
	using LinearAlgebra
end

# ╔═╡ 5cc88c68-9db3-4ec5-9262-f9fed684cdda
begin
	Plots.plot(10.0.^(range(-10, 10; length=256)/10), x -> pdf(source_prior, x), xscale=:log10)
	Plots.vline!(λ)
end

# ╔═╡ 7da18e2b-2303-4abe-9a78-3a960c562dc8
begin
    n_samples = 5000
    n_anneal  = 4

    #path  = ArithmeticPath(n_anneal)
    path  = GeometricPath(n_anneal)
    prop  = UniformNormalLocalProposal(0.0, 2.0)
    mcmc  = SliceSteppingOut([2.0, 2.0])

    jump   = IndepJumpProposal(prop)
    #jump   = AnnealedJumpProposal(prop, path)
end


# ╔═╡ 39931c63-9a60-4a78-94bc-61be95fd91f1
begin
    initial_params = WidebandDoA.WidebandIsoIsoParam{Float64}[]
    initial_order  = 0
	
    rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
    #rjmcmc = ReversibleJump.ReversibleJumpMCMC(order_prior, jump, mcmc)

    samples, stats = ReversibleJump.sample(
        rjmcmc,
        cond,
        n_samples,
        initial_order,
        initial_params;
        show_progress=false,
    )
end

# ╔═╡ 36e43974-dd74-4c11-a601-cde3178a1b8f
Plots.plot([stat.order for stat in stats], xlabel="RJMCMC Iteration", ylabel="Model order", label="Model order trace")

# ╔═╡ 75c9c639-7a09-4dbb-a8aa-d49695b2e5a0
Plots.histogram([stat.order for stat in stats], xlabel="order")

# ╔═╡ f0dcf695-7b9f-4929-8a48-be003ec8b4bc
begin
	k_post = [stat.order for stat in stats[n_samples ÷ 10:end]]
	ϕ_post = [
		[target.phi for target in sample]
		for sample in samples[n_samples ÷ 10:end]
	]
	k_mixture = round(Int, quantile(k_post, 0.8))
	mixture, labeled = WidebandDoA.relabel(Random.default_rng(), ϕ_post, k_mixture; show_progress=false)
end

# ╔═╡ 05e2cf48-8b23-4c94-8146-3f8fa4ac8def
begin
    flat = vcat(samples[n_samples ÷ 10:end]...)
    Plots.histogram([f.phi for f in flat], normed=true, bins=256, xlims=[-π/2, π/2], xlabel="DoA (ϕ)", label="Posterior")
    Plots.vline!(ϕ, label="True", color=:red, linestyle=:dash)
end

# ╔═╡ f20198ec-c064-4571-a983-83aab0e96eea
begin
    Plots.stephist([f.phi for f in flat], normed=true, bins=512, xlims=[-π/2, π/2], xlabel="DoA (ϕ)", label="Posterior", fill=true, alpha=0.5)
	Plots.plot!(range(-π/2,π/2; length=1024), Base.Fix1(pdf, mixture), label="Mixture Marginal Density", linewidth=2)
	Plots.vline!(ϕ, label="True", color=:red, linestyle=:dash)
end

# ╔═╡ bb716eb6-ecb2-4ce3-b76c-8f575d159ea4
begin
    Plots.stephist([f.phi for f in flat], normed=true, bins=512, xlims=[-π/2, π/2], xlabel="DoA (ϕ)", label="Posterior", fill=true)
	Plots.plot!(mixture, label=nothing)
	Plots.vline!(ϕ, label="True", color=:red, linestyle=:dash)
end

# ╔═╡ 290258ea-cb0b-420a-92a1-370732b76cd9
begin
	k = 3
	lambdas = [exp(sample[k].loglambda) for sample in samples[n_samples÷10:end]]
	Plots.stephist(lambdas, bins=64, label="SNR (λ) Posterior", fill=true)
	Plots.vline!(λ[k:k], label="True SNR", color=:red)
end

# ╔═╡ 2ef0c90f-393a-4748-a96b-34f0c71ab1ac
# ╠═╡ disabled = true
#=╠═╡
begin
	rng    = StableRNG(1)
    N      = 32
    M      = 20
    Δx     = range(0, M*0.5; length=M)
    c      = 1500
    fs     = 2000
    #ϕ      = [-π/4, 0.01, π/4]
    #ϕ      = [-2*π/5, -2.1*π/5, π/5, 2*π/5, 0.01]
    ϕ      = (-4:4)*π/11
    filter = WidebandDoA.WindowedSinc(N)
    λ      = fill(0.2, length(ϕ))
    #λ      = [3.0, 3.0, 3.0, 0.2, 3.0, 3.0, 3.0, 3.0]
    #λ      = [3.0, 0.2, 5.0]
    σ      = 1.0

	#α_λ, β_λ = 2.1, 0.6823408279481948
	#α_λ, β_λ = 0.01, 0.01

	#source_prior = InverseGamma(0.01, 0.01)
	source_prior = LogNormal(2.3^2, 2.3)

    order_prior = truncated(NegativeBinomial(1/2 + 0.1, 0.1/(0.1 + 1)), 0, M-1)
    model       = WidebandDoA.WidebandIsoIsoModel(
        N, Δx, c, fs, source_prior; order_prior, delay_filter=filter
    )

    params = rand(rng, model.prior; k=length(ϕ), sigma=σ, phi=ϕ, lambda=λ)
    y      = rand(rng, model.likelihood, model.prior, params.sourcesignals, ϕ; sigma=σ)
    #y     /= std(y)

    cond = WidebandConditioned(model, y)
end
  ╠═╡ =#

# ╔═╡ 56b27ce4-49a7-49ed-933f-99b94ccb43f6
begin
	include("common.jl")

	rng = Random.default_rng()
	ϕ   = collect(-4:4)*π/11
	snr = -5.0

	source_prior = LogNormal(2.3^2, 2.3)
	model, _ = construct_default_model(rng, ϕ, snr)
    cond    = @set model.model.prior.source_prior = source_prior
end

# ╔═╡ Cell order:
# ╠═71585db6-d5de-11ee-2f91-6306270cfdbb
# ╠═fd1c1453-8293-4371-9296-8b3b8baad279
# ╠═56b27ce4-49a7-49ed-933f-99b94ccb43f6
# ╠═2ef0c90f-393a-4748-a96b-34f0c71ab1ac
# ╠═5cc88c68-9db3-4ec5-9262-f9fed684cdda
# ╠═7da18e2b-2303-4abe-9a78-3a960c562dc8
# ╠═39931c63-9a60-4a78-94bc-61be95fd91f1
# ╠═36e43974-dd74-4c11-a601-cde3178a1b8f
# ╠═75c9c639-7a09-4dbb-a8aa-d49695b2e5a0
# ╠═f0dcf695-7b9f-4929-8a48-be003ec8b4bc
# ╠═05e2cf48-8b23-4c94-8146-3f8fa4ac8def
# ╠═f20198ec-c064-4571-a983-83aab0e96eea
# ╠═bb716eb6-ecb2-4ce3-b76c-8f575d159ea4
# ╠═290258ea-cb0b-420a-92a1-370732b76cd9
