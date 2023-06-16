from tueplots import bundles

from ._tueplots_bundles import beamer_moml as beamer_moml_bundle, jmlr as jmlr_bundle

_tueplots_bundles = {
    "imprs_2022": bundles.beamer_moml,
    "jmlr": jmlr_bundle,
    "research_project": jmlr_bundle,
    "thesis_talk": beamer_moml_bundle,
    "uk_2023": beamer_moml_bundle,
}

_colors = {
    "jmlr": {
        "u": "C0",
        "sol": "C1",
        "bc": "C2",
        "pde": "C3",
        "u_meas": "C4",
    },
    "uk_2023": {
        "u": "C0",
        "sol": "C1",
        "bc": "C2",
        "pde": "C3",
        "u_meas": "C4",
    },
}
