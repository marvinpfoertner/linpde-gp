from tueplots import bundles, figsizes, fonts, fontsizes


def jmlr(*, rel_width=1.0, nrows=1, ncols=1, family="serif", **kwargs):
    size = figsizes.jmlr2001(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        **kwargs,
    )
    font_config = fonts.jmlr2001_tex(family=family)
    fontsize_config = fontsizes.jmlr2001()
    return {**font_config, **size, **fontsize_config}


def beamer_moml(*, rel_width=1.0, rel_height=0.8, nrows=1):
    return bundles.beamer_moml(
        rel_width=rel_width,
        rel_height=rel_height,
    )
