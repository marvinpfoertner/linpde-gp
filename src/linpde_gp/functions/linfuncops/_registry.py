from linpde_gp import functions, linfuncops

########################################################################################
# `SelectOutput` #######################################################################
########################################################################################


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, f: functions.StackedFunction, /):
    assert isinstance(self.idx, int)
    assert f.output_ndim == 1

    return f.fns[self.idx]
