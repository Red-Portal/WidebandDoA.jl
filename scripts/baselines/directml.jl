
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
    n_eval_point  ::Int,
    rate_upsample ::Int,
    visualize     ::Bool,
)
    θ′ = barycentric_linesearch(
        1, 
        n_eval_point, 
        rate_upsample; 
        visualize=visualize
    ) do θ_range
        map(θi -> dml_loglikelihood(vcat(θ, θi), R, n_snapshots, f_range, conf), θ_range)
    end
    vcat(θ, θ′)
end

function dml_greedy_optimize(
    n_sources,
    R,
    n_snapshots,
    f_range,
    conf;
    n_eval_point,
    rate_upsample,
    visualize
)
    θ = Float64[]
    for k in 1:n_sources
        θ = dml_incremental_optimize(
            θ,
            R,
            n_snapshots,
            f_range,
            conf;
            n_eval_point,
            rate_upsample,
            visualize,
        )
    end
    θ
end

function dml_sage(
    y,
    R,
    n_sources,
    f_range,
    conf;
    n_iters  ::Int      = 100,
    θ_init              = nothing,
    n_eval_point ::Int  = 256,
    rate_upsample::Int  = 128,
    tolerance    ::Real = 1e-4,
    visualize    ::Bool = false,
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
        dml_greedy_optimize(
            n_sources, R, N, f_range, conf;
            n_eval_point, rate_upsample, visualize
        )
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
    for i in 1:n_iters
        for k in 1:n_sources
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

            θ[k] = barycentric_linesearch(
                Base.Fix1(map, cond_objective),
                1, 
                n_eval_point, 
                rate_upsample; 
                visualize=false,
            ) |> only

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

        if i > 1 && abs(loglike[i] - loglike[i-1]) < tolerance
            return θ, loglike[i]
        end

        if visualize
            Plots.plot(loglike[1:i]) |> display
        end
    end
    θ, last(loglike)
end
