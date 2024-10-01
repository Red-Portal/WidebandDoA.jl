
# [Inference](@id inference)

We provide more details on the inference functionalities provided by `WidebandDoA` and `ReversibleJump`.


## RJMCMC Samplers

Currently, `ReversibleJump` provides two RJMCMC samplers:
```julia
ReversibleJumpMCMC(order_prior, jump_proposal, mcmc_kernel; jump_rate)
NonReversibleJumpMCMC(jump_proposal, mcmc_kernel; jump_rate)
```
`ReversibleJumpMCMC` is the birth-death-update reversible jump MCMC sampler proposed by Green [^G1995], while `NonReversibleJumpMCMC` is the non-reversible counterpart described in the paper, and originally proposed by Gagnon and Doucet[^GD2020].

[^G1995]: Green, Peter J. "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination." Biometrika 82.4 (1995): 711-732.
[^GD2020]: Gagnon, Philippe, and Arnaud Doucet. "Nonreversible jump algorithms for Bayesian nested model selection." Journal of Computational and Graphical Statistics 30.2 (2020): 312-323.

### Arguments
- `order_prior`: A prior on the model order.
- `jump_proposal`: A jump proposal provided by WidebandDoA.
- `mcmc_kernel`: An RJMCMC sampler provided by `ReversibleJump`.


### Keyword Arguments
- `jump_rate::Real`: Upper bound on the probabiliy of proposing a jump move.


## Jump Proposals

We only provide a single option for the jump proposals:

```@docs; canonical=false
WidebandDoA.UniformNormalLocalProposal
```
This should be wrapped with

```
    IndepJumpProposal(prop)
```

This can then be used as follows:
```julia
jump   = IndepJumpProposal(prop)
rjmcmc = ReversibleJumpMCMC(order_prior, jump, mcmc)
```

## MCMC Samplers

### Slice Samplers

For the update move, we provide the following selection of MCMC samplers:

```@docs; canonical=false
Slice
SliceSteppingOut
SliceDoublingOut
```
For more information, refer to the original paper by Neal[^N2003].

[^N2003]: Neal, Radford M. "Slice sampling." The annals of statistics 31.3 (2003): 705-767.

!!! info
    For sampling from `WidebandIsoIsoModel`, the *first* element of the window is used for the DoA parameter $\phi$, and the *last* element is used for $\log\lambda$. Therefore, use the following: `window = [phi_window, loglambda_window]`.

For example, this can be used as followS:

```julia
mcmc   = SliceSteppingOut([2.0, 2.0]),
rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
```

### Metropolis-Hastings Samplers


We also provide Metropolis-Hastings samplers:

```@docs; canonical=false
RandomWalkMetropolis
IndependentMetropolis
MetropolisMixture
```
To use these, however, the following wrapper has to be used:
```@docs; canonical=false
WidebandIsoIsoMetropolis
```

For example, this can be used as followS:

```julia
mcmc = WidebandIsoIsoMetropolis(
    MetropolisMixture(
        IndependentMetropolis(Uniform(-π/2, π/2)),
        RandomWalkMetropolis(0.5)
    ),
    RandomWalkMetropolis(0.5)
)
rjmcmc = ReversibleJump.NonReversibleJumpMCMC(jump, mcmc; jump_rate=0.9)
```
