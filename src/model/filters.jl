
function inter_sensor_delay(ϕ::AbstractVector, Δx::AbstractVector, c)
    Tullio.@tullio avx=64 threads=false τ[m,k] := Δx[m]*sin(ϕ[k])/c
end


"""
    WindowedSinc(n_fft)

S. -C. Pei and Y. -C. Lai, "Closed Form Variable Fractional Time Delay Using FFT," 
in IEEE Signal Processing Letters, 2012.
    
S. -C. Pei and Y. -C. Lai, 
"Closed form variable fractional delay using FFT with transition band trade-off," 
IEEE International Symposium on Circuits and Systems (ISCAS), 2014.

Note: Technically speaking this filter is now overkill.
"""
struct WindowedSinc <: AbstractDelayFilter
    n_fft
end

function array_delay(filter::WindowedSinc, Δn::Matrix{T})  where {T<:Real}
    n_fft = filter.n_fft
    θ     = collect(0:n_fft-1)*2*π/n_fft
    a_fd  = T(0.25)
    Tullio.@tullio avx=64 threads=false H[n,m,k] := begin
        if (n - 1) <= floor(Int, n_fft/2)
            if (n - 1) == 0
                Complex{T}(1.0)
            elseif (n - 1) <= ceil(Int, n_fft/2) - 2
                exp(-1im*Δn[m,k]*θ[n])
            elseif (n - 1) <= ceil(Int, n_fft/2) - 1
                a_fd*cos(Δn[m,k]*π) + (1 - a_fd)*exp(-1im*Δn[m,k]*2*π/n_fft*(n_fft/2 - 1))
            elseif (n - 1) == ceil(Int, n_fft/2)
                Complex{T}(cos(Δn[m,k]*π))
            end
        else
            zero(Complex{T})
        end
    end
    idx_begin_cplx = ceil(Int, n_fft/2) + 1
    H[idx_begin_cplx:end,:,:] = begin
        if isodd(n_fft)
            conj.(H[idx_begin_cplx-1:-1:2,:,:])
        else
            conj.(H[idx_begin_cplx:-1:2,:,:])
        end
    end
    H
end

struct ComplexShift <: AbstractDelayFilter
    n_fft
end

function array_delay(filter::ComplexShift, Δn::Matrix{T}) where {T <: Real}
    n_fft = filter.n_fft
    ω     = collect(0:n_fft-1)*2*π/n_fft
    Tullio.@tullio H[n,m,k] := exp(-1im*Δn[m,k]*ω[n])
end
