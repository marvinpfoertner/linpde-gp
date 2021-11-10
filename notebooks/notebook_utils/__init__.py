import dataclasses
import pathlib
from collections.abc import Iterable

from . import _setup  # pylint: disable=unused-import


@dataclasses.dataclass()
class Config:
    notebook_name: str = None

    debug_mode: bool = False

    _results_path: pathlib.Path = None
    _notebook_results_path: pathlib.Path = None

    _savefig_default_extensions = (".pdf",)

    @property
    def results_path(self) -> pathlib.Path:
        if self._results_path is None:
            self._results_path = pathlib.Path(__file__).parents[2] / "results"
            self._results_path.mkdir(exist_ok=True)

            if self.debug_mode:
                self._results_path /= "debug"
                self._results_path.mkdir(exist_ok=True)

        return self._results_path

    @property
    def notebook_results_path(self) -> pathlib.Path:
        if self._notebook_results_path is None:
            if self.notebook_name is None:
                raise ValueError(
                    "Must set `config.notebook_name` in order to use `notebook_utils`!"
                )

            self._notebook_results_path = (
                self.results_path / "notebooks" / self.notebook_name
            )
            self._notebook_results_path.mkdir(parents=True, exist_ok=True)

        return self._notebook_results_path

    @property
    def savefig_default_extensions(self) -> tuple[str]:
        return self._savefig_default_extensions

    @savefig_default_extensions.setter
    def savefig_default_extensions(self, extensions: Iterable[str]) -> None:
        self._savefig_default_extensions = tuple(extensions)


config = Config()


from ._savefig import savefig

__all__ = [
    "config",
    "savefig",
]
