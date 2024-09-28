
"""
    UniformNormalLocalProposal(mu, sigma)

Uniform proposal over the DoAs and a log-normal propsal over the SNR parameters of a source.

# Arguments
* `mu`: Mean of the log-normal proposal on the SNR parameter.
* `sigma`: Standard deviation of the log-normal proposal on the SNR parameter.

This corresponds to the following proposal:

```math
\\begin{aligned}
    \\phi_j &\\sim q\\left(\\phi\\right) = \\mathsf{Uniform}\\left(-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right) \\\\
    \\gamma_j &\\sim q\\left(\\gamma\\right) = \\text{\\sf{}Log-Normal}\\left(\\text{\\tt{}mu}, \\text{\\tt{}sigma}\\right)
\\end{aligned}
```

"""
struct UniformNormalLocalProposal{F <: Real}
    mu   ::F
    sigma::F
end
