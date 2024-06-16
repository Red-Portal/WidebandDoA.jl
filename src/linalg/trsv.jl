
function trsv_striped_matrix!(A::AbstractArray, x::Array)
    @assert size(A,2) == size(A,3)
    @assert size(A,2) == size(x,2)
    @assert size(A,1) == size(x,1)
    B   = size(A,1)
    N   = size(A,2)
    buf = zeros(eltype(A), B)
    for i in 1:N
        Ai = view(A, 1:B, i, 1:i-1)
        xi = view(x, 1:B,    1:i-1)
        @tullio buf[b] = Ai[b,k].*xi[b,k] 
        x[:,i] = (x[:,i] - buf)./A[:,i,i]
    end
    x
end
