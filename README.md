[![PyPI version](https://img.shields.io/pypi/v/bletl)](https://pypi.org/project/bletl)
[![pipeline](https://github.com/jubiotech/bletl/workflows/pipeline/badge.svg)](https://github.com/jubiotech/bletl/actions)
[![coverage](https://codecov.io/gh/jubiotech/bletl/branch/main/graph/badge.svg)](https://codecov.io/gh/jubiotech/bletl)
[![documentation](https://readthedocs.org/projects/bletl/badge/?version=latest)](https://bletl.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5101434.svg)](https://doi.org/10.5281/zenodo.5101434)

# Installation
`bletl` is available through [PyPI](https://pypi.org/project/bletl/):

```
pip install bletl
```

## For Developers
You can use `bletl` by installing it in your Python environment.
1. clone it `git clone https://github.com/JuBiotech/bletl`
2. `cd bletl`
3. `pip install -e .` to install it into your (activated!) Python environment in "editable mode"

# Contributing
The easiest way to contribute is to report bugs by opening [Issues](https://github.com/JuBiotech/bletl/issues).

If you want to contribute, you should...
1. Clone `bletl`
2. Create a new branch
3. Make changes on your feature-branch
4. Open a [Pull Request](https://github.com/JuBiotech/bletl/pulls)


# Usage and Citing
`bletl` is licensed under the [GNU Affero General Public License v3.0](https://github.com/JuBiotech/bletl/blob/main/LICENSE.md).

When using `bletl` in your work, please cite the [Osthege & Tenhaef et al. (2022) paper](https://doi.org/10.1002/elsc.202100108) __and__ the [corresponding software version](https://doi.org/10.5281/zenodo.5101434).

Note that the paper is a shared first co-authorship, which can be indicated by <sup>1</sup> in the bibliography.

```bibtex
@article{bletlPaper,
  author   = {Osthege$^1$, Michael and
              Tenhaef$^1$, Niklas and
              Zyla, Rebecca and
              Müller, Carolin and
              Hemmerich, Johannes and
              Wiechert, Wolfgang and
              Noack, Stephan and
              Oldiges, Marco},
  title    = {bletl - A Python package for integrating {B}io{L}ector microcultivation devices in the {D}esign-{B}uild-{T}est-{L}earn cycle},
  journal  = {Engineering in Life Sciences},
  volume   = {22},
  number   = {3-4},
  pages    = {242-259},
  keywords = {BioLector, feature extraction, growth rate, microbial phenotyping, uncertainty quantification},
  doi      = {https://doi.org/10.1002/elsc.202100108},
  url      = {https://onlinelibrary.wiley.com/doi/abs/10.1002/elsc.202100108},
  eprint   = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/elsc.202100108},
  year     = {2022}
}

@software{bletl,
  author       = {Michael Osthege and
                  Niklas Tenhaef and
                  Laura Helleckes and
                  Carolin Müller},
  title        = {JuBiotech/bletl: v1.1.0},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.6284777},
  url          = {https://doi.org/10.5281/zenodo.6284777}
}
```

Head over to Zenodo to [generate a BibTeX citation](https://doi.org/10.5281/zenodo.5101434) for the latest release.
