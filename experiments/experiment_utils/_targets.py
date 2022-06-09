from tueplots import bundles

from ._tueplots_bundles import beamer_moml as beamer_moml_bundle, jmlr as jmlr_bundle

_tueplots_bundles = {
    "imprs_2022": bundles.beamer_moml,
    "jmlr": jmlr_bundle,
    "research_project": jmlr_bundle,
    "thesis_talk": beamer_moml_bundle,
}
