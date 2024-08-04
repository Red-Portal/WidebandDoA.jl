
function criterion(::Val{:aic}, ℓ, p, n)
    -2*ℓ + 2*p
end

function criterion(::Val{:mdl}, ℓ, p, n)
    -ℓ + p/2*log(n)
end
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
    n_snapshots     ::Int,
    f_range         ::AbstractVector,
    conf            ::ArrayConfig;
    n_am_iterations ::Int  = 10,
    n_eval_point    ::Int  = 256,
    rate_upsample   ::Int  = 8,
    visualize       ::Bool = false,
)
    
    n_bins    = size(R, 3)
    n_channel = size(R, 2)

    @assert n_max_targets < n_channel

    # Find Targets
    θs       = [Float64[]]
    loglikes = [dml_loglikelihood(last(θs), R, n_snapshots, f_range, conf)]

    for m in 1:n_max_targets
        θ_init = dml_incremental_optimize(
            last(θs),
            R,
            n_snapshots,
            f_range,
            conf;
            n_eval_point,
            rate_upsample,
            visualize,
        )
        θ, loglike = dml_alternating_maximization(
            R,
            m,
            n_snapshots,
            f_range,
            conf;
            visualize,
            θ_init,
            n_iterations=n_am_iterations,
            n_eval_point,
            rate_upsample,
        )
        push!(θs, θ)
        push!(loglikes, loglike)
    end

    ks   = 0:n_max_targets
    crit = map(ks) do k
        ℓ = loglikes[k+1]

        # number of parameters (implicitly) inferred through ML is:
        #     degree of freedom =  source signal + doas + noise variance .
        # See P.-J. Chung, M. Viberg, J. Yu, 2013, Ch 14.
        p = 2*n_bins*n_snapshots*k + k + 1
        n = n_bins*n_snapshots

        criterion(Val(crit), ℓ, n, p)
    end

    if visualize
        Plots.plot(0:n_max_targets, crit) |> display
    end
    k = ks[argmin(crit)]
    k, θs[k+1]
end

# function infocrit(
#     crit       ::Symbol,
#     R,
#     n_snapshots::Int;
#     visualize  ::Bool = false,
# )
#     n_channels = size(R, 2)
#     n_bins     = size(R, 3)

#     λ = mapreduce(hcat, 1:n_bins) do n
#         Rω = view(R, :, :, n)
#         λω, _ = eigen(Rω)
#         @. max(real(λω), eps(Float64))
#     end

#     ks   = 0:n_channels - 1
#     crit = map(ks) do k
#         ℓ = sum(1:n_bins) do n
#             λ_noise = view(λ, 1:n_channels - k, n)
#             n_snapshots*(n_channels - k)*log(
#                 StatsBase.geomean(λ_noise) / mean(λ_noise)
#             )
#         end
#         p = n_bins*k*(2*n_channels - k)
#         n = n_snapshots*n_bins

#         criterion(Val(crit), ℓ, p, n)
#     end
#     if visualize
#         Plots.plot(0:n_channels - 1, crit) |> display
#     end
#     ks[argmin(crit)]
# end
