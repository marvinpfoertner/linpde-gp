# LinPDE-GP: Linear PDE Solvers based on GP Regression

Code for the Paper "Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers"

## Getting Started

### Cloning the Repository

This repository includes Git submodules, so it is best cloned via

```shell
git clone --recurse-submodules git@github.com:marvinpfoertner/linpde-gp.git
```

If you forgot the `--recurse-submodules` flag when cloning, simply run

```shell
git submodule update --init --recursive
```

inside the repository.

### Installing a Full Development Environment

```shell
cd path/to/linpde-gp
pip install -r dev-requirements.txt
```
