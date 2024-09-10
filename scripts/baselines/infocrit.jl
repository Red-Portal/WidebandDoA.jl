
function criterion(::Val{:aic}, ℓ, p, n)
    -2*ℓ + 2*p
end

function criterion(::Union{Val{:mdl}, Val{:bic}}, ℓ, p, n)
    -ℓ + p/2*log(n)
end

function infocrit(
    crit            ::Symbol,
    loglikes        ::AbstractVector,
    θs              ::AbstractVector,
    n_bins          ::Int,
    n_snapshots     ::Int;
    visualize       ::Bool = false,
)
    ks   = 0:length(loglikes)-1
    crit = map(ks) do k
        ℓ = loglikes[k+1]

        # number of parameters (implicitly) inferred through ML is:
        #     degree of freedom =  source signal + doas + noise variance .
        # See P.-J. Chung, M. Viberg, J. Yu, 2013, Ch 14.
        p = 2*n_bins*n_snapshots*k + k + n_bins
        n = n_bins*n_snapshots

        criterion(Val(crit), ℓ, p, n)
    end

    if visualize
        Plots.plot(0:length(loglikes)-1, crit) |> display
    end
    k = ks[argmin(crit)]
    k, θs[k+1]
end

function infocrit(
    crit         ::Symbol,
    y,
    R,
    n_max_sources::Int,
    f_range      ::AbstractVector,
    conf         ::ArrayConfig;
    n_iters      ::Int  = 100,
    tolerance    ::Real = 1e-6,
    visualize    ::Bool = false,
)
    n_bins  = size(R, 3)
    n_snap  = size(y, 1)
    θs, lls = dml_sequential_ml(y, R, n_max_sources, f_range, conf; n_iters, visualize, tolerance)
    infocrit(crit, lls, θs, n_bins, n_snap; visualize)
end

