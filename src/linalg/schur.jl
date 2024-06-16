
function inv_hermitian_striped_matrix!(S)
    #
    # In-place inversion of hermitian striped matrices
    #
    # A. Ladaycia, A. Mokraoui, K. Abed-Meraim, and A. Belouchrani, 
    # "Performance bounds analysis for semi-blind channel estimation in 
    # MIMO-OFDM communications systems,"
    # IEEE Trans. Wireless Commun. 2017.
    #
    # The striped matrix is stored in the dense-A format S ∈ R^{ N × M × M}

    @assert( size(S,2) == size(S,3) )

    M = size(S,2)
    N = size(S,1)
    ϵ = eps(S |> eltype |> real)

    Tullio.@tullio threads=false ℓdetS := sum(log(abs(S[i,1,1])) + ϵ)
    Tullio.@tullio threads=false S[i,1,1] = safe_complex_reciprocal(S[i,1,1], ϵ)
    
    X⁻¹Y_buf         = Array{eltype(S)}(undef, N, M, 1)
    YᴴX⁻¹_buf        = Array{eltype(S)}(undef, N, 1, M)
    X⁻¹YZ⁻¹_buf      = Array{eltype(S)}(undef, N, M, 1)
    YᴴX⁻¹Y           = Array{eltype(S)}(undef, N, 1, 1)
    X⁻¹YZ⁻¹YᴴX⁻¹_buf = Array{eltype(S)}(undef, N, M, M)

    for m = 1:M-1
        X⁻¹ = view(S,:,1:m,1:m) 
        Y   = view(S,:,1:m,m+1:m+1)
        Yᴴ  = view(S,:,m+1:m+1,1:m)
        Z   = view(S,:,m+1:m+1,m+1:m+1)

        X⁻¹Y         = view(X⁻¹Y_buf,         :, 1:m, 1:1)
        YᴴX⁻¹        = view(YᴴX⁻¹_buf,        :, 1:1, 1:m)
        X⁻¹YZ⁻¹      = view(X⁻¹YZ⁻¹_buf,      :, 1:m, 1:1)
        X⁻¹YZ⁻¹YᴴX⁻¹ = view(X⁻¹YZ⁻¹YᴴX⁻¹_buf, :, 1:m, 1:m)
        Z⁻¹          = Z

        #X⁻¹Y = X⁻¹⊛Y
        stripe_matmul!(X⁻¹, Y, X⁻¹Y)

        #YᴴX⁻¹Y = Yᴴ ⊛ X⁻¹Y
        stripe_matmul!(Yᴴ, X⁻¹Y, YᴴX⁻¹Y)

        Tullio.@tullio threads=false Z⁻¹[i,1,1] = safe_complex_reciprocal(Z[i,1,1] - YᴴX⁻¹Y[i,1,1], ϵ)
        Tullio.@tullio threads=false ℓdetSdivE := -log(abs(Z⁻¹[i,1,1]) + ϵ)

        Tullio.@tullio threads=false error := Z⁻¹[i,1,1]*(Z[i,1,1] - YᴴX⁻¹Y[i,1,1])

        if !isfinite(ℓdetSdivE) || abs(error)/size(Z,1) > 5
            return nothing, -Inf
        end

        #X⁻¹YZ⁻¹ = X⁻¹Y⊛Z⁻¹
        stripe_matmul!(X⁻¹Y, Z⁻¹, X⁻¹YZ⁻¹)

        #YᴴX⁻¹
        stripe_matmul!(Yᴴ, X⁻¹, YᴴX⁻¹)

        # X⁻¹YZ⁻¹YᴴX⁻¹ = X⁻¹YZ⁻¹ ⊛ YᴴX⁻¹ = X⁻¹YZ⁻¹ ⊛ (X⁻¹Y)ᴴ
        stripe_matmul!(X⁻¹YZ⁻¹, YᴴX⁻¹, X⁻¹YZ⁻¹YᴴX⁻¹)

        ℓdetS += ℓdetSdivE

        # Notes:
        # 1. Tullio doesn't seem to work with in-place block structures
        # 2. Loopvectorization has the smallest memory footprint
        # 3. The order of operation below matters for both performance and exactness
        @turbo warn_check_args=false for i in 1:size(S,1)
            S[i,m+1,m+1] = Z⁻¹[i,1,1]
        end
        @turbo warn_check_args=false for k in 1:m, j in 1:m, i in 1:size(S,1)
            S[i,j,k]     = X⁻¹[i,j,k] + X⁻¹YZ⁻¹YᴴX⁻¹[i,j,k]
        end
        @turbo warn_check_args=false for j in 1:m, i in 1:size(S,1)
            S[i,j,m+1]   = -X⁻¹YZ⁻¹[i,j,1]
            S[i,m+1,j]   = conj(S[i,j,m+1])
        end
    end
    S, ℓdetS
end
