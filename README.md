
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](http://krkim.me/WidebandDoA.jl/)
[![Build Status](https://github.com/Red-Portal/WideBandDOA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Red-Portal/WideBandDOA.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Bayesian Wideband Direction-of-Arrival Estimation with RJMCMC

This repository provides the code to reproduce the paper:
> 

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

Then, you should be able to run the tests
```julia
using Pkg
Pkg.test("WidebandDoA")
```





