
module WidebandDoA

using Distributions
using LinearAlgebra
using Random 
using Statistics

using LoopVectorization
using Tullio

#include("inference/slice.jl")
#include("inference/imhrwmh.jl")
include("linalg/striped_matrix.jl")

abstract type AbstractDelayFilter end

"""
    array_delay(filter, Δn)

Returns the fourier domain fractional delay filters as a matrix
    H ∈ R^{ N × M × K }
The fractional delay filters are the ones in:
"""
function array_delay end

include("models/filters.jl")
#include("models/normalgamma.jl")

end
