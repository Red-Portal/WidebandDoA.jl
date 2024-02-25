
function rformat(D)
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

function aformat(A, n_blocks, blocksize)
    D         = Array{eltype(A)}(undef, blocksize, n_blocks, n_blocks)
    idx_block = collect(partition(1:blocksize*n_blocks, blocksize))
    for m = 1:n_blocks, n = 1:n_blocks
        D[:,m,n] = diag(A[idx_block[m], idx_block[n]]) 
    end
    D
end

@testset "striped matrix" begin
    realtype  = Float64
    blocksize = 128
    n_blocks  = 10
    D_real    = randn(realtype, blocksize, n_blocks, n_blocks)
    D_cplx    = randn(realtype, blocksize, n_blocks, n_blocks)
    D         = Complex.(D_real, D_cplx)

    for n = 1:n_blocks
        D[:,n,n] = abs.(D[:,n,n])
        for m = n+1:n_blocks
            D[:,n,m] = conj(D[:,m,n])
        end
    end

    A = rformat(D) |> Hermitian

    idx_block = collect(partition(1:blocksize*n_blocks, blocksize))

    norm_A = norm(A)
    @testset "R format" begin
        @assert ishermitian(A)
        @assert(norm(aformat(A, n_blocks, blocksize) - D) < norm_A*1e-3)
        @assert norm(diag(A[idx_block[1],idx_block[2]]) - D[:,1,2]) < norm_A*1e-3
        @assert norm(diag(A[idx_block[2],idx_block[2]]) - D[:,2,2]) < norm_A*1e-3
    end

    @testset "inv_hermitian_striped_matrix" begin
        D⁻¹        = deepcopy(D)
        D⁻¹, ℓdetA = WidebandDoA.inv_hermitian_striped_matrix!(D⁻¹)

        A⁻¹ = rformat(D⁻¹)

        @testset "inverse" begin
            @test norm(A⁻¹*A - I)    < norm_A*1e-3
            @test norm(inv(A)*A - I) < norm_A*1e-3
        end

        @testset "log determinant" begin
            ℓdetA_true = logabsdet(A)[1]
            @test ℓdetA ≈ ℓdetA_true rtol=1e-3
        end
    end
end
