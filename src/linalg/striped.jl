

function rformat(D::AbstractArray)
    @assert size(D,2) == size(D,3)
    n_blocks  = size(D, 2)
    blocksize = size(D, 1)
    A         = Matrix{eltype(D)}(undef, n_blocks*blocksize, n_blocks*blocksize)
    idx_block = collect(partition(1:blocksize*n_blocks, blocksize))
    for m = 1:n_blocks, n = 1:n_blocks
        A[idx_block[m], idx_block[n]] = diagm(D[:,m,n])
    end
    A
end

function aformat(A::AbstractArray, n_blocks::Int, blocksize::Int)
    D         = Array{eltype(A)}(undef, blocksize, n_blocks, n_blocks)
    idx_block = collect(partition(1:blocksize*n_blocks, blocksize))
    for m = 1:n_blocks, n = 1:n_blocks
        D[:,m,n] = diag(A[idx_block[m], idx_block[n]]) 
    end
    D
end

@inline function ⊛(A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3})
    @assert size(A,3) == size(B,2)
    @assert size(A,1) == size(B,1)
    Tullio.@tullio threads=false C[n,a,b] := A[n,a,c]*B[n,c,b]
end

@inline function stripe_matmul!(A::AbstractArray{<:Any,3}, 
                                B::AbstractArray{<:Any,3},
                                C::AbstractArray{<:Any,3})
    @assert size(A,3) == size(B,2)
    @assert size(A,1) == size(B,1)
    @assert size(C,2) == size(A,2)
    @assert size(C,3) == size(B,3)
    Tullio.@tullio threads=false C[n,a,b] = A[n,a,c]*B[n,c,b]
end

@inline function stripe_adjoint(A::AbstractArray{<:Any,3})
    Tullio.@tullio threads=false Aᴴ[n,b,a] := conj(A[n,a,b])
end

@inline function stripe_adjoint!(A::AbstractArray{<:Any,3}, Aᴴ::AbstractArray{<:Any,3})
    Tullio.@tullio threads=false Aᴴ[n,b,a] = conj(A[n,a,b])
end

@inline function safe_complex_reciprocal(x, ϵ)
    a     = real(x)
    b     = imag(x)
    x_mag = a*a + b*b 
    Complex(a/x_mag, -b/x_mag)
end

