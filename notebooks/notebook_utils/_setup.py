import pathlib

import jax
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

# Use custom matplotlib style file
plt.style.use(pathlib.Path(__file__).parent / "linpde-gp.mplstyle")

# Set output formats for matplotlib inline backend
set_matplotlib_formats("svg")

# Jax configuration
jax.config.update("jax_enable_x64", True)
