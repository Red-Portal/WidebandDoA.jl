
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

    log(invlike_nul) - log(invlike_alt)
end

function null_statistics(
    n_snapshots::Int,
    n_channel  ::Int,
    m          ::Int,
)
    n1 = n_snapshots*(2 + 1)
    n2 = n_snapshots*(2*n_channel - 2*m - 1)

    μ  = digamma(n1/2 + n2/2) - digamma(n2/2)
    σ2 = trigamma(n2/2)       - trigamma(n1/2 + n2/2)
    μ, σ2
end

function boostrap_statistics(
    rng             ::Random.AbstractRNG,
    z               ::AbstractVector,
    n_bootstrap     ::Int,
    n_bootstrap_nest::Int,
    μ_null          ::Real,
)
    n_bins = length(z)
    map(1:n_bootstrap) do _
        z_boot      = StatsBase.sample(rng, z, n_bins)
        μ_boot      = mean(z_boot)
        μ_boot_nest = map(1:n_bootstrap_nest) do _
            mean(StatsBase.sample(rng, z_boot, n_bins))
        end
        σ_nest = std(μ_boot_nest; corrected=true)
        abs(μ_boot - μ_null) / σ_nest
    end
end

function test_threshold(
    z      ::AbstractVector,
    μ_null ::Real,
    σ2_null::Real,
)
    n_bins = length(z)
    abs(mean(z) - μ_null) / sqrt(σ2_null)
end

function benjaminihochberg(
    pvals    ::AbstractVector,
    q        ::Real,
    M        ::Int,
    visualize::Bool
)
    orders_sorted = sortperm(pvals)
    pvals_sorted  = pvals[orders_sorted]

    k = 0
    for m in 1:M
        if pvals_sorted[m] ≤ q*m/M
            k = m
        end
    end

    if visualize
        Plots.plot(sort(pvals_sorted)) |> display
        Plots.plot!(1:M, m -> q*m/M) |> display
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
    Y,
    R,
    rate_false_detection::Real,
    n_max_targets       ::Int,
    n_snapshots         ::Int,
    f_range             ::AbstractVector,
    conf                ::ArrayConfig;
    n_bootstrap        = 256,
    n_bootstrap_nest   = 256,
    n_ml_iterations    = 200,
    ml_inner_tolerance = 1e-6,
    ml_outer_tolerance = 1e-6,
    visualize          = true,
)
    #=
        P. Chung, J. F. Bohme, C. F. Mecklenbrauker and A. O. Hero, 
        "Detection of the Number of Signals Using the Benjamini-Hochberg Procedure," 
        in IEEE Transactions on Signal Processing, 2007.
    =##

    n_channel = size(R, 2)

    @assert n_max_targets < n_channel

    p_values = Float64[]
    θs       = [Float64[]]

    for m in 1:n_max_targets
        θ     = last(θs)
        θ_alt = dml_incremental_optimize(
            θ,
            R,
            n_snapshots,
            f_range,
            conf;
            visualize,
            tolerance = ml_inner_tolerance,
        )

        if !isfinite(dml_loglikelihood(θ_alt, R, n_snapshots, f_range, conf))
            break
        end

        z = map(enumerate(f_range)) do (n, fc)
            ratio_test_statistic(θ_alt, θ, view(R, :,:,n), fc, conf)
        end

        μ_null, σ2_null = null_statistics(n_snapshots, n_channel, m)
        T_boot          = boostrap_statistics(rng, z, n_bootstrap, n_bootstrap_nest, μ_null)
        T_thres         = test_threshold(z, μ_null, σ2_null)
        p_value         = mean(T_boot .> T_thres)

        if visualize
            @info(
                "",
                mean(z)        = mean(z),
                test_statistic = median(T_boot),
                test_threshold = T_thres,
                p_value
            )
        end

        θ_alt, _ = dml_sage(
            Y, R, m, f_range, conf;
            n_iters         = n_ml_iterations,
            θ_init          = θ_alt,
            inner_tolerance = ml_inner_tolerance,
            outer_tolerance = ml_outer_tolerance,
            visualize,
        )

        push!(p_values, p_value)
        push!(θs, θ_alt)
    end
    k = benjaminihochberg(
        p_values,
        rate_false_detection,
        length(p_values),
        visualize
    )
    k, θs[k+1], p_values
end
