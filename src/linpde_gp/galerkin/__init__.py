from . import _project_operator, bases


def discretize(bvp, basis: bases.Basis):
    import probnum as pn

    from ._project_rhs import project as project_rhs

    A = bvp.diffop.project(basis)
    b = project_rhs(bvp.rhs, basis)

    return pn.problems.LinearSystem(
        pn.linops.aslinop(A),
        b,
    )
