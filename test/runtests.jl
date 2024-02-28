
using Test
using WidebandDoA

using AbstractMCMC
using Base.Iterators
using Distributions
using FFTW
using LinearAlgebra
using MCMCTesting
using Random
using ReversibleJump
using StableRNGs
using Statistics
using Tullio

include("filters.jl")
include("striped.jl")
include("normalgamma.jl")
