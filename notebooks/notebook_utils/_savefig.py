from collections.abc import Iterable
from typing import Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt

from . import config


def savefig(
    filename: str,
    extension: Optional[Union[str, Iterable[str]]] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    **savefig_kwargs,
) -> None:
    if fig is None:
        fig = plt.gcf()

    if extension is None:
        extensions = config.savefig_default_extensions
    elif isinstance(extension, str):
        extensions = extension
    else:
        extensions = extension

    for extension in extensions:
        if extension.startswith("."):
            fname_ext = filename + extension
        else:
            fname_ext = f"{filename}.{extension}"

        fig.savefig(config.notebook_results_path / fname_ext, **savefig_kwargs)
