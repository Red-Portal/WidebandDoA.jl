
function dct_upsample(y, α)
    # The normalizations are adjusted to match the paper
    @assert( 2 ≤ α )
    P     = length(y)
    cₚ     = FFTW.dct(y)/sqrt(P)
    n_pad = round(Int, P*(α - 1))
    c_pad = vcat(cₚ, zeros(n_pad))
    g̃     =  FFTW.idct(c_pad)*sqrt(length(c_pad))
    g̃
end

function barycentric_weight_node(τ, g₂, w)
    # J. Selva, 
    # "Efficient Wideband DOA Estimation Through Function Evaluation Techniques," 
    # in IEEE Transactions on Signal Processing, 2018.
    #
    # Eq. (24)

    P       = length(g₂)
    Q       = div(length(w),2)
    n_near  = round(Int, (2*P*τ + 1)/2)
    t_near  = (2*n_near - 1)/(2*P)
    u       = P*(τ - t_near)
    q       = -Q:Q
    n_fetch = q .+ n_near
    inbound = 1 .< n_fetch .< P
    q       = q[inbound]
    n_fetch = n_fetch[inbound]
    w_itpl  = map(enumerate(q)) do (i, qᵢ)
        if qᵢ == 0
            1.0
        else
            u/(u + qᵢ)*w[i]*((-1)^qᵢ) + eps(Float64)
        end
    end
    w_itpl = w_itpl / sum(w_itpl)
    n_fetch, w_itpl
end

function eval_chebyschev_s(τ, g₂, w)
    n_fetch, w_itpl = barycentric_weight_node(τ, g₂, w)
    dot(g₂[n_fetch], w_itpl)
end

function eval_chebyschev_s′(τ, g₂, t, w)
    # J. Selva, 
    # "Design of Barycentric Interpolators for Uniform and Nonuniform Sampling Grids," 
    # in IEEE Transactions on Signal Processing, 2010.
    #
    # Eq. (8)
    # s′′(τ) ≈ ∑ₙ (s(tₙ) - s(τ))/(tₙ - τ)⋅aₙ

    n_fetch, w_itpl = barycentric_weight_node(τ, g₂, w)

    g₃τ = eval_chebyschev_s(τ, g₂, w)
    dot((g₂[n_fetch] .- g₃τ)./(t[n_fetch] .- τ .+ eps(Float64)), w_itpl)
end

function eval_chebyschev_s′′(τ, g₂, t, w)
    # J. Selva, 
    # "Design of Barycentric Interpolators for Uniform and Nonuniform Sampling Grids," 
    # in IEEE Transactions on Signal Processing, 2010.
    #
    # Eq. (10)
    # s′′(τ) ≈ ∑ₙ 2!/(tₙ - τ)^2 ( s(tₙ) - ∑_p∈{0,1} s⁽²⁾(τ)/p!⋅(tₙ - τ)ᵖ )⋅aₙ
    #        = ∑ₙ 2!/(tₙ - τ)^2 ( s(tₙ) - s(τ) - s′(τ)⋅(tₙ - τ) )⋅aₙ

    n_fetch, w_itpl = barycentric_weight_node(τ, g₂, w)

    g₂t     = g₂[n_fetch]
    g₃τ     = eval_chebyschev_s(τ, g₂, w)
    g₃′τ    = eval_chebyschev_s′(τ, g₂, t, w)
    t_fetch = t[n_fetch]
    dot(2 ./ (t_fetch .- τ .+ eps(Float64)).^2 .* (g₂t .- g₃τ - g₃′τ*(t_fetch .- τ)), w_itpl)
end

function coarse_optimization(g₂::AbstractVector, n_targets::Int)
    # J. Selva, 
    # "Efficient Wideband DOA Estimation Through Function Evaluation Techniques," 
    # in IEEE Transactions on Signal Processing, 2018.
    #
    # Op 1
    
    P       = length(g₂)
    t       = (2*(1:P) .- 1) / (2*P)
    n_crit  = Peaks.argmaxima(g₂)
    t_crit  = t[n_crit]
    g₂_crit = g₂[n_crit]

    idx_crit_sort = sortperm(g₂_crit, rev=true)
    t_crit_sort   = t_crit[idx_crit_sort]
    t_targets     = t_crit_sort[1:n_targets]
    t_targets
end

function fine_optimization(t_coarse::AbstractVector, 
                           g₂::AbstractVector, 
                           rate_upsample::Int,
                           n_targets::Int)
    # J. Selva, 
    # "Efficient Wideband DOA Estimation Through Function Evaluation Techniques," 
    # in IEEE Transactions on Signal Processing, 2018.
    #
    # Op 3

    P = length(g₂)
    Q = rate_upsample*2
    q = -Q:Q
    α = rate_upsample
    w = abs.(sinc.((1 - 1/α).*sqrt.(Complex.(q.^2 .- (Q+1)^2))) ./ sinc(im*(1 - 1/α)*(Q + 1)))
    t = (2*(1:P) .- 1) / (2*P)

    t_opt = t_coarse
    for i = 1:n_targets
        for fuck = 1:3
            s′tᵢ  = eval_chebyschev_s′(t_opt[i], g₂, t, w)
            s′′tᵢ = eval_chebyschev_s′′(t_opt[i], g₂, t, w)

            t_opt[i] -= s′tᵢ / s′′tᵢ
        end
    end
    t_opt
end

function barycentric_linesearch(doa_spectrum::Function,
                                n_targets::Int,
                                n_eval_point::Int,
                                rate_upsample::Real;
                                visualize=true)
    # J. Selva, 
    # "Efficient Wideband DOA Estimation Through Function Evaluation Techniques," 
    # in IEEE Transactions on Signal Processing, 2018.

    n_eval_point_up = n_eval_point*rate_upsample

    tₙ = (2*(1:n_eval_point) .- 1) / (2*n_eval_point)
    θ  = cos.(π*tₙ) * π/2
    P  = doa_spectrum(θ)

    if visualize
        display(Plots.plot(θ, P, mark=:circ, label="Spectrum"))
    end

    t = (2*(1:n_eval_point_up) .- 1) / (2*n_eval_point_up)
    θ = cos.(π*t) * π/2
    P = dct_upsample(P, rate_upsample)

    if visualize
        display(Plots.plot!(θ, P, label="Upsampled Spectrum"))
    end

    t_coarse_opt = coarse_optimization(P, n_targets)
    t_fine_opt   = fine_optimization(t_coarse_opt, P, rate_upsample, n_targets)
    t_fine_opt   = t_coarse_opt
    θ_opt        = cos.(π*t_fine_opt) * π/2

    if visualize
        display(Plots.vline!(θ_opt))
    end
    θ_opt
end
