
function ldl_striped_matrix!(A::Array, ϵ = eps(real(eltype(A))))
    @assert size(A,2) == size(A,3)
    B = size(A,1)
    N = size(A,2)
    D = zeros(eltype(A), B, N)

    buf = zeros(eltype(A), B)

    @inbounds for j in 1:N
        Lj = view(A,1:B,j,1:j-1)
        Dj = view(D,1:B,1:j-1)
        @tullio buf[b] = abs2(Lj[b,k])*Dj[b,k]
        D[:,j] = A[:,j,j] - buf

        A[:,j,j] .= one(eltype(A))

        if any(@. abs(D[:,j]) ≤ ϵ)
            nothing, nothing
        end

        @inbounds for i in j+1:N
            Li = view(A,1:B,i,1:j-1)
            @tullio buf[b] = Li[b,k]*conj(Lj[b,k])*Dj[b,k]
            A[:,i,j] = (A[:,i,j] - buf) ./ D[:,j]
        end
    end
    @inbounds for j in 1:N, i in 1:j-1
        A[:,i,j] .= zero(eltype(A))
    end
    D, A
end

function cholesky_striped_matrix!(A::Array, ϵ = eps(real(eltype(A))))
    @assert size(A,2) == size(A,3)
    B = size(A,1)
    N = size(A,2)
    buf = zeros(eltype(A), B)

    for j in 1:N
        Lj = view(A,1:B,j,1:j-1)
        @tullio buf[b] = abs2(Lj[b,k])
        A[:,j,j] = sqrt.(A[:,j,j] - buf)

        if any(@. abs(A[:,j,j]) ≤ ϵ)
            nothing
        end
        
        @inbounds for i in j+1:N
            Li = view(A,1:B,i,1:j-1)
            @tullio buf[b] = conj(Lj[k])*Li[k]
            A[:,i,j] = (A[:,i,j] - buf)./A[:,j,j]
        end
    end
    @inbounds for j in 1:N, i in 1:j-1
        A[:,i,j] .= zero(eltype(A))
    end
    A
end
