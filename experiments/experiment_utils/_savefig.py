from collections.abc import Iterable

import matplotlib.figure
import matplotlib.pyplot as plt

from . import config


def savefig(
    filename: str,
    extension: str | Iterable[str] | None = None,
    fig: matplotlib.figure.Figure | None = None,
    **savefig_kwargs,
) -> None:
    if fig is None:
        fig = plt.gcf()

    if extension is None:
        extensions = config.savefig_default_extensions
    elif isinstance(extension, str):
        extensions = [extension]
    else:
        extensions = extension

    for extension in extensions:
        if extension.startswith("."):
            fname_ext = filename + extension
        else:
            fname_ext = f"{filename}.{extension}"

        fig.savefig(config.experiment_results_path / fname_ext, **savefig_kwargs)
