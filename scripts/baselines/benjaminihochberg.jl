
function loglikelihood(
    θ      ::AbstractVector,
    R, 
    f_range::AbstractVector,
    conf   ::ArrayConfig
)
    sum(enumerate(f_range)) do (n, fc)
        P    = proj(θ, fc, conf)
        P⊥  = I - P
        Rω   = view(R,:,:,n)
        -log(real(tr(P⊥*Rω)))
    end
end

function ratio_test_statistic(
    θ_alt  ::AbstractVector,
    θ_nul  ::AbstractVector,
    R_bin  ::AbstractMatrix,
    fc     ::Real,
    conf   ::ArrayConfig
)
    P_alt = proj(θ_alt, fc, conf)
    P_nul = if length(θ_nul) == 0
        Zeros(size(R_bin))
    else
        proj(θ_nul, fc, conf)
    end

    P⊥_alt = I - P_alt
    P⊥_nul = I - P_nul

    -log(real(tr(P⊥_alt*R_bin))) + log(real(tr(P⊥_nul*R_bin)))
end

function boostrap_statistics(
    rng             ::Random.AbstractRNG,
    z               ::AbstractVector,
    n_bootstrap     ::Int,
    n_bootstrap_nest::Int,
    n_bins          ::Int
)
    μ_est = mean(z)
    map(1:n_bootstrap) do _
        z_boot      = StatsBase.sample(rng, z, n_bins)
        μ_boot      = mean(z_boot)
        μ_boot_nest = map(1:n_bootstrap_nest) do _
            mean(StatsBase.sample(rng, z_boot, n_bins))
        end
        σ_boot = std(μ_boot_nest; corrected=true)
        abs(μ_boot - μ_est) / σ_boot
    end
end

function test_statistic(
    z               ::AbstractVector,
    n_temp_snapshots::Int,
    n_channel       ::Int,
    m               ::Int,
)
    μ_est = mean(z)
    n1    = n_temp_snapshots*(2 + 1)
    n2    = n_temp_snapshots*(2*n_channel - 2*m - 1)
    μ     = digamma(n1/2 + n2/2) - digamma(n2/2)
    σ2    = trigamma(n2/2)       - trigamma(n1/2 + n2/2)
    abs(μ_est - μ) / sqrt(σ2)
end

function benjaminihochberg(
    pvals::AbstractVector,
    q    ::Real,
    M    ::Int
)
    orders_sorted = sortperm(pvals)
    pvals_sorted  = pvals[orders_sorted]

    k = 0
    for m in 1:M
        if pvals_sorted[m] ≤ q*m/M
            k = m
        else
            break
        end
    end

    if k == 0
        return 0
    else
        orders_rejected = orders_sorted[1:k]
        maximum(orders_rejected)
    end
end

function likeratiotest(
    rng                 ::Random.AbstractRNG,
    R,
    rate_false_detection::Real,
    n_max_targets       ::Int,
    n_temp_snapshots    ::Int,
    f_range             ::AbstractVector,
    conf                ::ArrayConfig;
    n_bootstrap      = 1024,
    n_bootstrap_nest = 1024,
    n_eval_point     = 64,
    rate_upsample    = 8,
    visualize        = false
)
    #=
        P. Chung, J. F. Bohme, C. F. Mecklenbrauker and A. O. Hero, 
        "Detection of the Number of Signals Using the Benjamini-Hochberg Procedure," 
        in IEEE Transactions on Signal Processing, 2007.
    =##

    n_bins    = size(R, 3)
    n_channel = size(R, 2)

    @assert n_max_targets < n_channel

    p_values = zeros(n_max_targets)
    θ        = Float64[]
    for m in 1:n_max_targets
        θ′ = barycentric_linesearch(
            1, 
            n_eval_point, 
            rate_upsample; 
            visualize=visualize
        ) do θ_range
            map(θi -> loglikelihood(vcat(θ, θi), R, f_range, conf), θ_range)
        end
        θ_alt = vcat(θ, θ′)

        z = map(enumerate(f_range)) do (n, fc)
            ratio_test_statistic(θ_alt, θ, view(R, :,:,n), fc, conf)
        end

        T_boot  = boostrap_statistics(rng, z, n_bootstrap, n_bootstrap_nest, n_bins)
        T       = test_statistic(z, n_temp_snapshots, n_channel, m)
        rank    = sum(T_boot .< T)
        p_value = 1 - rank/n_bootstrap

        p_values[m] = p_value
        θ           = θ_alt
    end
    k = benjaminihochberg(
        p_values, rate_false_detection, n_max_targets
    )
    k, k == 0 ? nothing : θ[1:k]
end
