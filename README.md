# Binary distribution in the Grassmann formalism
Implementation of binary distribution in the Grassmann formalism, including conditional version.
The Grassmann formalism was introduced in [1].

See the [pdf](https://github.com/mackelab/grassmann_binary_distribution/blob/main/notes_grassmann_formalism.pdf)  file for more explanations.

@JCBrouwer: This fork is for me to get familiar with the Grassmann distribution by cleaning up the code (mainly reducing code duplication and vectorizing things) plus extending support for categorical and ordinal distributions as described in [2].

## File structure
- grassmann_distribution/:
  - GrassmannDistribution: definition of Grassmann (gr) as well as mixture of Grassmann (mogr) distribution
  - fit_grassmann: corresponding functions to estimate a gr / mogr (moment matching as well as MLE)
  - conditional_grassmann: implements a conditional mogr in the same spirit as a MDN for a MoGauss
- notebooks/: some example notebooks how to define the distributions and an example to fit a mogr to dichotomized gauss data
- data: samples for a dichotomized gauss distribution, see [3] for details.
- tests/: can be run with `pytest`

## Install the package:

This fork:
`pip install git+https://github.com/JCBrouwer/grassmann`

The original (needed to run tests):
`pip install git+https://github.com/mackelab/grassmann_binary_distribution`

## References

[1] Arai, T. (2021). Multivariate binary probability distribution in the Grassmann formalism. Physical Review E, 103(6), 062104.

[2] Arai, T. (2023). Multivariate probability distribution for categorical and ordinal random variables. Preprint, arXiv, 2304.00617.

[3] Macke, J. H., et al. (2009). Generating spike trains with specified correlation coefficients. Neural computation 21.2: 397-423.
