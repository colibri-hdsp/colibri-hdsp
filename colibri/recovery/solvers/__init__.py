from .spc import SPCSolver


SOLVERS = {
    "SPC": SPCSolver
}


def get_solver(acquisition_model):
    return SOLVERS[acquisition_model.__class__.__name__]