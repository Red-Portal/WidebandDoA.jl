
function proj(θ::AbstractVector, fc::Real, conf::ArrayConfig)
    n_channels = length(conf.Δx)
    if length(θ) == 0
        Zeros(eltype(θ), n_channels, n_channels)
    else
        A = steering_matrix(θ, fc, conf)
        A*pinv(A)
    end
end

function loglikelihood(
    θ      ::AbstractVector,
    R, 
    f_range::AbstractVector,
    conf   ::ArrayConfig
)
    sum(enumerate(f_range)) do (n, fc)
        try
            P     = proj(θ, fc, conf)
            P⊥   = I - P
            Rω    = view(R,:,:,n)
            -real(tr(P⊥*Rω))
        catch
            -Inf
        end
    end
end
