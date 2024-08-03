
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

    invlike_alt = real(tr(P⊥_alt*R_bin))
    invlike_nul = real(tr(P⊥_nul*R_bin))

    -log(invlike_alt) + log(invlike_nul)
end

function null_statistics(
    n_temp_snapshots::Int,
    n_channel       ::Int,
    m               ::Int,
)
    n1    = n_temp_snapshots*(2 + 1)
    n2    = n_temp_snapshots*(2*n_channel - 2*m - 1)
    μ     = digamma(n1/2 + n2/2) - digamma(n2/2)
    σ2    = trigamma(n2/2)       - trigamma(n1/2 + n2/2)
    μ, σ2
end

function boostrap_statistics(
    rng             ::Random.AbstractRNG,
    z               ::AbstractVector,
    n_temp_snapshots::Int,
    n_channel       ::Int,
    m               ::Int,
    n_bootstrap     ::Int,
    n_bootstrap_nest::Int,
    n_bins          ::Int,
)
    μ, _  = null_statistics(n_temp_snapshots, n_channel, m)
    map(1:n_bootstrap) do _
        z_boot      = StatsBase.sample(rng, z, n_bins)
        μ_boot      = mean(z_boot)
        μ_boot_nest = map(1:n_bootstrap_nest) do _
            mean(StatsBase.sample(rng, z_boot, n_bins))
        end
        σ_nest = std(μ_boot_nest; corrected=true, mean=μ_boot)
        abs(μ_boot - μ) / σ_nest
    end
end

function test_threshold(
    z               ::AbstractVector,
    n_temp_snapshots::Int,
    n_channel       ::Int,
    m               ::Int,
)
    μ_est = mean(z)
    μ, σ2 = null_statistics(n_temp_snapshots, n_channel, m)
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
    n_bootstrap      = 128,
    n_bootstrap_nest = 128,
    n_eval_point     = 1024,
    rate_upsample    = 8,
    visualize        = true,
)
    #=
        P. Chung, J. F. Bohme, C. F. Mecklenbrauker and A. O. Hero, 
        "Detection of the Number of Signals Using the Benjamini-Hochberg Procedure," 
        in IEEE Transactions on Signal Processing, 2007.
    =##

    n_bins    = size(R, 3)
    n_channel = size(R, 2)

    @assert n_max_targets < n_channel

    p_values = ones(n_max_targets)
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

        if !isfinite(loglikelihood(θ_alt, R, f_range, conf))
            break
        end

        z = map(enumerate(f_range)) do (n, fc)
            ratio_test_statistic(θ_alt, θ, view(R, :,:,n), fc, conf)
        end

        T_boot  = boostrap_statistics(
           rng,
           z,
           n_temp_snapshots,
           n_channel,
           m,
           n_bootstrap,
           n_bootstrap_nest,
           n_bins
        )
        T_thres = test_threshold(z, n_temp_snapshots, n_channel, m)
        rank    = sum(T_boot .< T_thres)
        p_value = 1 - rank/n_bootstrap

        if visualize
            @info(
                "",
                test_statistic = mean(T_boot),
                test_threshold = T_thres,
                rank           = rank,
                p_value
            )
        end

        p_values[m] = p_value
        θ           = θ_alt
    end
    k = benjaminihochberg(
        p_values, rate_false_detection, n_max_targets
    )
    k, k == 0 ? Float64[] : θ[1:k]
end

function likeratiotest(
    R,
    rate_false_detection::Real,
    n_max_targets       ::Int,
    n_temp_snapshots    ::Int,
    f_range             ::AbstractVector,
    conf                ::ArrayConfig;
    n_bootstrap      = 128,
    n_bootstrap_nest = 128,
    n_eval_point     = 1024,
    rate_upsample    = 8,
    visualize        = true,
)
   likeratiotest(
       Random.default_rng(),
       R,
       rate_false_detection,
       n_max_targets,
       n_temp_snapshots,
       f_range,
       conf;
       n_bootstrap      = n_bootstrap,
       n_bootstrap_nest = n_bootstrap_nest,
       n_eval_point     = n_eval_point,
       rate_upsample    = rate_upsample,
       visualize        = visualize,
    ) 
end
