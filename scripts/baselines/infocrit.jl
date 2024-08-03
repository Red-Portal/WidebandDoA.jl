
function criterion(::Val{:aic}, ℓ, n, p)
    -2*ℓ + 2*p
end

function criterion(::Val{:mdl}, ℓ, n, p)
    -ℓ + p/2*log(n)
end

function infocrit(
    crit            ::Symbol,
    R,
    n_max_targets   ::Int,
    n_temp_snapshots::Int,
    f_range         ::AbstractVector,
    conf            ::ArrayConfig;
    n_eval_point    ::Int    = 1024,
    rate_upsample   ::Int    = 8,
    visualize       ::Bool   = false,
)
    
    n_bins    = size(R, 3)
    n_channel = size(R, 2)

    @assert n_max_targets < n_channel

    # Find Targets
    θ = Float64[]
    for _ in 1:n_max_targets
        θ′ = barycentric_linesearch(
            1, 
            n_eval_point, 
            rate_upsample; 
            visualize=visualize
        ) do θ_range
            map(θi -> loglikelihood(vcat(θ, θi), R, f_range, conf), θ_range)
        end
        θ = vcat(θ, θ′)
    end

    ks   = 0:n_max_targets
    crit = map(ks) do k
        θk = k == 0 ? Float64[] : θ[1:k]
        ℓ  = loglikelihood(θk, R, f_range, conf)

        # number of parameters (implicitly) inferred through ML is:
        #     degree of freedom =  source signal + doas + noise variance .
        # See P.-J. Chung, M. Viberg, J. Yu, 2013, Ch 14.
        p = 2*n_bins*n_temp_snapshots*k + k + 1
        n = n_temp_snapshots

        criterion(Val(crit), ℓ, n, p)
    end

    if visualize
        Plots.plot(0:n_max_targets, crit) |> display
    end
    k = ks[argmin(crit)]

    k, k == 0 ? Float64[] : θ[1:k]
end
