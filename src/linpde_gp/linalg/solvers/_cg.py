import probnum as pn
import scipy.sparse
import scipy.sparse.linalg


class ConjugateGradients:
    def solve(self, linear_system: pn.problems.LinearSystem, **cg_kwargs):
        if isinstance(linear_system.A, pn.linops.LinearOperator):
            A = scipy.sparse.linalg.LinearOperator(
                shape=linear_system.A.shape,
                dtype=linear_system.A.dtype,
                matvec=lambda vec: linear_system.A @ vec,
            )
        else:
            A = linear_system.A

        (x, _) = scipy.sparse.linalg.cg(
            A,
            linear_system.b,
            **cg_kwargs,
        )

        return pn.randvars.Constant(support=x)
