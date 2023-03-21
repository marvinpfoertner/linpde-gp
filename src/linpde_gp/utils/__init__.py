try:
    # `matplotlib` is an optional dependency
    from . import plotting
except ModuleNotFoundError as exc:
    if "matplotlib" not in exc.msg:  # pylint: disable=unsupported-membership-test
        raise exc
