
using Tullio

@testset "striped matrix" begin
    realtype  = Float64
    blocksize = 3
    n_blocks  = 2
    S         = zeros(Complex{realtype}, 3, 2, 2)
    S[:,1,1]  = [1, 2, 3]
    S[:,2,2]  = [1, 2, 3]
    S[:,1,2]  = fill(realtype(0.1)*im, 3)
    S[:,2,1]  = fill(realtype(-0.1)*im, 3)

    A = WidebandDoA.rformat(S) |> Hermitian

    idx_block = collect(partition(1:blocksize*n_blocks, blocksize))

    norm_A = norm(A)
    @testset "R format" begin
        @assert ishermitian(A)
        @assert(norm(WidebandDoA.aformat(A, n_blocks, blocksize) - S) < 1e-7)
        @assert norm(diag(A[idx_block[1],idx_block[2]]) - S[:,1,2]) < 1e-7
        @assert norm(diag(A[idx_block[2],idx_block[2]]) - S[:,2,2]) < 1e-7
    end

    @testset "inv_hermitian_striped_matrix" begin
        S⁻¹        = deepcopy(S)
        S⁻¹, ℓdetA = WidebandDoA.inv_hermitian_striped_matrix!(S⁻¹)

        A⁻¹ = WidebandDoA.rformat(S⁻¹)

        @testset "inverse" begin
            @test norm(A⁻¹*A - I) < 1e-7
        end

        @testset "log determinant" begin
            ℓdetA_true = logabsdet(A)[1]
            @test ℓdetA ≈ ℓdetA_true atol=1e-7
        end
    end

    @testset "cholesky_striped_matrix" begin
        L = WidebandDoA.cholesky_striped_matrix!(deepcopy(S))

        @testset "decomposition" begin
            @tullio LLt[b,i,j] := L[b,i,k]*conj(L[b,j,k])
            @test norm(S -  LLt) < 1e-7
        end

        @testset "log determinant" begin
            ℓdetA_true = logabsdet(A)[1]

            Tullio.@tullio threads=false ℓdetA := 2*log(abs(L[n,k,k]))

            @test ℓdetA ≈ ℓdetA_true atol=1e-7
        end
    end

    @testset "ldl_striped_matrix" begin
        D, L = WidebandDoA.ldl_striped_matrix!(deepcopy(S))

        @testset "decomposition" begin
            @tullio LDLt[b,i,j] := L[b,i,k]*D[b,k]*conj(L[b,j,k])
            @test norm(S -  LDLt) < 1e-7
        end

        @testset "log determinant" begin
            ℓdetA_true = logabsdet(A)[1]

            Tullio.@tullio threads=false ℓdetA := log(abs(D[n,k]))

            @test ℓdetA ≈ ℓdetA_true atol=1e-7
        end
    end

    @testset "trsv_striped_matrix" begin
        @testset "cholesky" begin
            L = WidebandDoA.cholesky_striped_matrix!(deepcopy(S))

            @testset "forward substitution" begin
                x_true = randn(eltype(S), blocksize, n_blocks)
                @tullio b[n,i] := L[n,i,j]*x_true[n,j]
        
                x = WidebandDoA.trsv_striped_matrix!(L, deepcopy(b))

                @test norm(x - x_true) < 1e-7
            end

            # @testset "quadratic form" begin
            #     x         = randn(eltype(S), blocksize, n_blocks)
            #     x_flat    = reshape(x_true, :)
            #     L_true    = cholesky(A)
            #     L⁻¹x_true = L_true \ x_flat

            #     L⁻¹x  = WidebandDoA.trsv_striped_matrix!(L, deepcopy(x))
            #     xS⁻¹x = sum(abs2, L⁻¹x)

            #     sum(abs2, L⁻¹x_true)
            # end
        end

        @testset "ldl" begin
            D, L = WidebandDoA.ldl_striped_matrix!(deepcopy(S))

            @testset "quadratic form" begin
                x          = randn(eltype(S), blocksize, n_blocks)
                x_flat     = reshape(x, :)
                xS⁻¹x_true = abs(x_flat'*(A \ x_flat))

                L⁻¹x  = WidebandDoA.trsv_striped_matrix!(L, deepcopy(x))
                @tullio xS⁻¹x := abs(L⁻¹x[n,i]/D[n,i]*conj(L⁻¹x[n,i]))

                @test xS⁻¹x ≈ xS⁻¹x_true atol=1e-5
            end
        end
    end
end
