
function reconstruct(
    model ::WidebandIsoIsoModel,
    params::AbstractVector{<:WidebandIsoIsoParam},
)
    @unpack y, y_fft, y_power, prior = model
    @unpack delay_filter, Δx, c, fs, alpha, beta, alpha_lambda, beta_lambda, order_prior = prior
    @unpack alpha, beta = prior

    # Σ_post  = (W⁻¹ + HᴴH)⁻¹
    # μ_post  = Σx_post Hᴴy
    # α_post  = α + NM/2
    # β_post  = β + 1/2( yᴴy + μx_postᴴ W μx_post)
    #         = β + 1/2( yᴴy - yᴴH(W⁻¹ + HᴴH)⁻¹Hᴴy )
    #         = β + 1/2 yᴴ(I - H(W⁻¹ + HᴴH)⁻¹Hᴴ)y

    k = length(params)
    ϕ = [θj.phi            for θj in params]
    λ = [exp(θj.loglambda) for θj in params]

    N = size(y_fft,1)
    M = size(y_fft,2)

    τ   = inter_sensor_delay(ϕ, Δx, c)
    H_S = array_delay(delay_filter, τ*fs)

    Tullio.@tullio HᴴH_S[n,j,k]     := conj(H_S[n,m,j]) * H_S[n,m,k]
    Tullio.@tullio W⁻¹pHᴴH_S[n,j,k] := (j == k) ? HᴴH_S[n,j,k] + 1/λ[k] : HᴴH_S[n,j,k]
    Tullio.@tullio Hᴴy_S[n,k]       := conj(H_S[n,m,k]) * y_fft[n,m]

    Σ_post_S, _ = Wideband.inv_hermitian_striped_matrix!(W⁻¹pHᴴH_S)

    Tullio.@tullio threads=false μ_post_S[n,k] := Σ_post_S[n,k,j] * Hᴴy_S[n,j]

    Φk, Φk⁻¹    = block_fft(k, N)
    Σ_post_S_R  = rformat(Σ_post_S)
    Σ_post      = PDMats.PDMat(
        Hermitian(
            real.(Φk⁻¹*(Σ_post_S_R*Φk))
        )
    )
    Hᴴy    = real.(ifft(Hᴴy_S,1))
    μ_post = vcat(real.(ifft(μ_post_S,1))...)*sqrt(N)
    α_post = alpha + N*M/2
    β_post = beta + (sum(abs2,y) - dot(Hᴴy,μ_post))/2

    MvTDist(2*α_post, μ_post, (β_post/α_post)*Σ_post)
end
