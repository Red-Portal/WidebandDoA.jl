
"""
    inter_sensor_delay(ϕ, Δx, c)

Compute the inter-sensor delay matrix \$D \\in \\mathbb{R}^{M \\times K}\$ in seconds for a linear array, where `M = length(isd)` is the number of sensors on the array, and `K = length(isd)` is the number of targets.
The matrix is computed as follows:
```math
{[D]}_{m,k} = \\frac{\\Delta x[m] \\, \\sin(\\phi[k])}{c}
```

# Arguments
* `ϕ::AbstractVector`: Vector of DoAs in radian. 
* `Δx::AbstractVector`: Vector of inter-sensor delay of the array.
* `c`: Propagation speed of the medium.

# Returns
* `delays`: Matrix containing the delays in seconds. Each row correspond to sensor, and each column correspond to the source.
"""
function inter_sensor_delay(ϕ::AbstractVector, Δx::AbstractVector, c)
    Tullio.@tullio threads=false τ[m,k] := Δx[m]*sin(ϕ[k])/c
end

"""
    WindowedSinc(n_fft) <: AbstractDelayFilter

Closed-form fractional delay filter by Pei and Lai[^PL2012][^PL2014]

# Arguments
* `n_fft::Int`: Number of taps of the filter.

[^PL2012]: S. -C. Pei and Y. -C. Lai, "Closed Form Variable Fractional Time Delay Using FFT," *IEEE Signal Processing Letters*, 2012.
[^PL2014]: S. -C. Pei and Y. -C. Lai, "Closed form variable fractional delay using FFT with transition band trade-off," In *Proceedings of the IEEE International Symposium on Circuits and Systems* (ISCAS), 2014.
"""
struct WindowedSinc <: AbstractDelayFilter
    n_fft::Int
end

function array_delay(filter::WindowedSinc, Δn::Matrix{T})  where {T<:Real}
    n_fft = filter.n_fft
    θ     = collect(0:n_fft-1)*2*T(π)/n_fft
    a_fd  = T(0.25)
    Tullio.@tullio H[n,m,k] := begin
        if (n - 1) <= floor(Int, n_fft/2)
            if (n - 1) == 0
                Complex{T}(1.0)
            elseif (n - 1) <= ceil(Int, n_fft/2) - 2
                exp(-1im*-Δn[m,k]*θ[n])
            elseif (n - 1) <= ceil(Int, n_fft/2) - 1
                a_fd*cos(-Δn[m,k]*T(π)) +
                    (1 - a_fd)*exp(-1im*-Δn[m,k]*2*T(π)/n_fft*(T(n_fft)/2 - 1))
            elseif (n - 1) == ceil(Int, n_fft/2)
                Complex{T}(cos(-Δn[m,k]*T(π)))
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
    n_fft::Int
end

function array_delay(filter::ComplexShift, Δn::Matrix{T}) where {T <: Real}
    n_fft = filter.n_fft
    ω     = collect(0:n_fft-1)*2*T(π)/n_fft
    Tullio.@tullio H[n,m,k] := exp(1im*Δn[m,k]*ω[n])
end
