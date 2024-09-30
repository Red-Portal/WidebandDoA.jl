
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](http://krkim.me/WidebandDoA.jl/)
[![Build Status](https://github.com/Red-Portal/WideBandDOA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/WideBandDOA.jl/actions/workflows/CI.yml?query=branch%3Amain)

# WidebandDoA

This repository provides the code to reproduce the paper:
> Fully Bayesian Wideband Direction-of-Arrival Estimation with RJMCMC

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Red-Portal/ReversibleJump.jl.git")
Pkg.add(url="https://github.com/Red-Portal/WidebandDoA.jl.git")
```

However, to run the unit tests, some additioanal work is needed.
First, go to the root director of `WidebandDoA`, and execute the following:
```julia
using Pkg
Pkg.activate("test")
Pkg.develop(url="https://github.com/Red-Portal/ReversibleJump.jl.git")
Pkg.develop(url="https://github.com/Red-Portal/MCMCTesting.jl.git")
```
This is necessary because Julia's test runner is not a fan of unregistered packages like `ReversibleJump` and `MCMCTesting`.
    
Then, you should be able to run the tests:
```julia
using Pkg
Pkg.test("WidebandDoA")
```


## Demonstration
We provide a comprehensive demonstration of the package in the [documentation](https://krkim.me/WidebandDoA.jl/dev/demonstration/).
For instance, for signals generated from `k = 4` sources with the following angle-frequency spectrum:

![](https://github.com/Red-Portal/WidebandDoA.jl/blob/gh-pages/dev/angle_frequency_spectrum_plot.svg) 

We show how to obtain estimate the model order through our Bayesian model:

![](https://github.com/Red-Portal/WidebandDoA.jl/blob/gh-pages/dev/model_order_hist.svg)

and the direction-of-arrivals:

![](https://github.com/Red-Portal/WidebandDoA.jl/blob/gh-pages/dev/doa_hist.svg)

One can also reconstruct the latent source signals as follows:

![](https://github.com/Red-Portal/WidebandDoA.jl/blob/gh-pages/dev/recon_mmse_comparison.svg)



