
@testset "filters" begin
    N = 512
    x = randn(N)
    X = fft(x)

    delays_int = [
        0  3  5;
        7 11 13
    ]
    delays = Float64.(delays_int)

    @testset for filter  in [
        WidebandDoA.WindowedSinc(N),
        WidebandDoA.ComplexShift(N)
    ]
        H = WidebandDoA.array_delay(filter, delays)
        @tullio Y[n,m,k] := H[n,m,k]*X[n]
        y = real.(ifft(Y, 1))

        for idx in CartesianIndices(y[1,:,:])
            m, k = Tuple(idx)
            Δn   = delays_int[m,k]
            err  = y[Δn+1:end,m,k] - x[1:end-Δn]
            @test sqrt(mean(abs2, err)) < 0.1
        end
    end
end
