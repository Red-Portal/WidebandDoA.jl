
# General Usage

We provide detailed documentation of `WidebandDoA` and `ReversibleJump` here.
A comprehensive example usage of these functionalities can be found in the [demo](@ref demo)

## Sampling

This package uses [`ReversibleJump`](https://github.com/Red-Portal/ReversibleJump.jl) for drawing samples.
To draw samples, the user needs to call the following function

```julia
ReversibleJump.sample([rng,] sampler, model, N, initial_order; initial_params, show_progress)
```
### Arguments
- `sampler`: An RJMCMC sampler provided by `ReversibleJump`.
- `model`: A model provided by `WidebandDoA` conditioned on data.
- `N::Int`: The number of samples.
- `initial_order::Int`: The model order of `initial_params`.

### Keyword Arguments
- `show_progress::Bool`: Whether to enable `ProgresMeter`
- `initial_params`: The initial state of the Markov chain. The order of this parameter must match `initial_order` for correct results.


For more details about the available options for `sampler`, refer to [here](@ref inference).

## Bayesian Model 
The Bayesian model described in the paper can be constructed through the following constructor:
```@docs; canonical=false
WidebandIsoIsoModel
```
`IsoIso` in the name indicates that we are using an isotrpoic normal prior on both the source signals and the noise.

This model can be conditioned on data by invoking the following constructor:
```@docs; canonical=false
WidebandDoA.WidebandConditioned
```

The overall process is as follows. Given a received signal `y`:
```julia
model = WidebandDoA.WidebandIsoIsoModel(
    N, Δx, c, fs, source_prior, alpha, beta; order_prior
)
cond  = WidebandConditioned(model, y)
```
This can then be used with `ReversibleJump.sample`.

## Simulating Data
Given a `model`, prior and likelihood simulations can be done through the following specialization of `Base.rand`:
```@docs; canonical=false
Base.rand
```

## Reconstruction
Given a conditioned model, reconstruction of the latent signals conditional on a set of parameters can be done through the following function:
```@docs; canonical=false
WidebandDoA.reconstruct
```

## Relabling
As explained in the paper, to analyze the individual sources, the RJMCMC samplers need to be relabeled. 
The labels and the Gaussian mixture approximation on the DoAs can be generated through the following function:
```@docs; canonical=false
WidebandDoA.relabel
```

