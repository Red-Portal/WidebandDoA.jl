
function proj(θ::AbstractVector, fc::Real, conf::ArrayConfig)
    n_channels = length(conf.Δx)
    if length(θ) == 0
        Zeros(eltype(θ), n_channels, n_channels)
    else
        A = steering_matrix(θ, fc, conf)
        A*pinv(A)
    end
end

function dml_loglikelihood(
    θ          ::AbstractVector,
    R, 
    n_snapshots::Int,
    f_range    ::AbstractVector,
    conf       ::ArrayConfig
)
    n_channels = size(R,1)
    sum(enumerate(f_range)) do (n, fc)
        try
            P   = proj(θ, fc, conf)
            P⊥ = I - P
            Rω  = view(R,:,:,n)
            σ2  = real(tr(P⊥*Rω))/n_channels
            -n_channels*n_snapshots*log(σ2)
        catch
            -Inf
        end
    end
end

function dml_incremental_optimize(
    θ             ::AbstractVector,
    R,
    n_snapshots   ::Int,
    f_range       ::AbstractVector,
    conf          ::ArrayConfig;
    visualize     ::Bool,
)
    obj(θk) = dml_loglikelihood(vcat(θ, θk), R, n_snapshots, f_range, conf)

    res = Optim.optimize(θk′ -> -obj(θk′), -π/2, π/2, Brent();)
    θk  = Optim.minimizer(res)

    res = Optim.optimize(
        θk′ -> -obj(θk′ |> only),
        [-π/2],
        [π/2],
        [θk],
        SAMIN(neps=size(R,1), verbosity=0),
        Optim.Options(
            time_limit=5,
            show_every=0,
            show_trace=false,
        )
    );
    if visualize
        display(res)
    end
    θk = Optim.minimizer(res) |> only

    res = Optim.optimize(
        θk′ -> -obj(θk′ |> only),
        [-π/2],
        [π/2],
        [θk],
        Fminbox(
            LBFGS(linesearch=LineSearches.BackTracking())
        ),
    );
    if visualize
        display(res)
    end
    θk = Optim.minimizer(res) |> only
    

    if visualize
        Plots.plot(range(-π/2, π/2; length=1024), obj) |> display
        Plots.vline!([θk]) |> display
    end
    vcat(θ, θk)
end

function dml_refine_optimize(
    θ             ::AbstractVector,
    R,
    n_snapshots   ::Int,
    f_range       ::AbstractVector,
    conf          ::ArrayConfig;
    visualize     ::Bool,
)
    k = length(θ)
    res = Optim.optimize(
        θ′ -> -dml_loglikelihood(θ′, R, n_snapshots, f_range, conf),
        fill(-π/2, k),
        fill(π/2, k),
        θ,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations=10,
            show_every=1,
        );
    )
    if visualize
        display(res)
    end
    Optim.minimizer(res), -Optim.minimum(res)
end

function dml_greedy_optimize(
    n_sources,
    R,
    n_snapshots,
    f_range,
    conf;
    visualize
)
    θ = Float64[]
    for _ in 1:n_sources
        θ = dml_incremental_optimize(θ, R, n_snapshots, f_range, conf; visualize)
    end
    θ
end

function dml_sage(
    y,
    R,
    n_sources,
    f_range,
    conf;
    n_iters  ::Int  = 100,
    θ_init          = nothing,
    visualize::Bool = false,
    tolerance::Real = 1e-6,
)
    #=
        Space-Alternating Generalized Expectation-Maximization (SAGE)
        for Deterministic Maximum Likelihood

        Fessler, Jeffrey A., and Alfred O. Hero.
        "Space-alternating generalized expectation-maximization algorithm."
        IEEE Transactions on signal processing 42.10 (1994): 2664-2677.

        Chung, Pei Jung, and Johann F. Bohme.
        "Comparative convergence analysis of EM and SAGE algorithms in DOA estimation."
        IEEE Transactions on Signal Processing 49.12 (2001): 2940-2949.
    =## 

    K = n_sources
    N = size(y,1)
    M = size(y,2)
    J = size(y,3)

    θ = if isnothing(θ_init)
        dml_greedy_optimize(n_sources, R, N, f_range, conf; visualize)
    else
        θ_init
    end

    # z ∈ C^{ M, N × J }
    # y ∈ C^{ N × M × J }
    # s ∈ C^{ K × N × J }
    # Rk ∈ C^{ M × M × J }

    s  = zeros(ComplexF64, K, N, J)
    A  = zeros(ComplexF64, M, K, J)
    ν  = ones(Float64, J)

    loglike = Array{Float64}(undef, n_iters)
    θ_opt  = deepcopy(θ)
    ll_opt = dml_loglikelihood(θ_opt, R, N, f_range, conf)
    for i in 1:n_iters
        for k in shuffle(1:n_sources)
            zk = zeros(ComplexF64, M, N, J)
            Rk = zeros(ComplexF64, M, M, J)

            # E-step
            @inbounds for j in 1:J
                f        = f_range[j]
                A[:,:,j] = steering_matrix(θ, f, conf)
            end

            @inbounds for j in 1:J
                Aj = view(A, :, :,   j)
                aj = view(A, :, k:k, j)
                yj = view(y, :, :, j)
                sj = view(s, :, :, j)

                zk[:,:,j] = aj*sj[k:k,:] + (transpose(yj) - Aj*sj)

                zkj = view(zk, :, :, j)
                Rk[:,:,j] = zkj*zkj'/M
            end

            # M-step
            cond_objective(θk) = begin
                sum(1:J) do j
                    f  = f_range[j]
                    aj = steering_matrix([θk], f, conf)[:,1]
                    real(aj'*view(Rk, :, :, j)*aj)
                end
            end

            res  = Optim.optimize(θk -> -cond_objective(θk), -π/2, π/2, Brent();)
            θ[k] = Optim.minimizer(res) |> only

            @inbounds for j in 1:J 
                f   = f_range[j]
                aj  = steering_matrix(θ[k:k], f, conf)[:,1]
                zkj = view(zk, :, :, j)
                Rkj = view(Rk, :, :, j)

                s[k,:,j] = conj.(zkj'*aj) / M
                ν[j]     = real(tr((I - aj*aj'/M)*Rkj))/M
            end
        end
        loglike[i] = dml_loglikelihood(θ, R, N, f_range, conf)

        if ll_opt < loglike[i]
            θ_opt  = θ
            ll_opt = loglike[i]
        end

        if i > 1 && abs(loglike[i] - loglike[i-1]) < tolerance
            return θ_opt, ll_opt
        end

        if visualize
            Plots.plot(loglike[1:i]) |> display
        end
    end
    θ_opt, ll_opt
end

function dml_alternating_maximization(
    y,
    R,
    n_sources,
    f_range,
    conf;
    n_iters  ::Int  = 50,
    θ_init          = nothing,
    visualize::Bool = false,
    tolerance::Real = 1e-6,
)
    K = n_sources
    N = size(y,1)

    θ = if isnothing(θ_init)
        dml_greedy_optimize(n_sources, R, N, f_range, conf; visualize)
    else
        θ_init
    end

    loglike = Array{Float64}(undef, n_iters)
    for t in 1:n_iters
        for k in 1:K
            obj(θk) = begin
                θ[k] = θk
                dml_loglikelihood(θ, R, N, f_range, conf)
            end
            res = Optim.optimize(θk -> -obj(θk), -π/2, π/2, Brent();)
            θ[k] = Optim.minimizer(res)
        end

        loglike[t] = dml_loglikelihood(θ, R, N, f_range, conf)

        if t > 1 && abs(loglike[t] - loglike[t-1]) < tolerance
           return θ, loglike[t]
        end

        if visualize
            Plots.plot(loglike[1:t]) |> display
        end
    end
    θ, last(loglike)
end
