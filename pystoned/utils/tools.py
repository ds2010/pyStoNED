# import dependencies
from re import compile
from os import environ
from pyomo.opt import SolverFactory, SolverManagerFactory
from ..constant import CET_ADDI, CET_MULT, OPT_LOCAL, OPT_DEFAULT
__email_re = compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')


def set_neos_email(address):
    """pass email address to NEOS server 

    Args:
        address (String): your own vaild email address.
    """
    if address == OPT_LOCAL:
        print("Optimizing locally.")
        return False
    if not __email_re.match(address):
        raise ValueError("Invalid email address.")
    environ['NEOS_EMAIL'] = address
    return True


def optimize_model(model, email, cet, solver=OPT_DEFAULT):
    if not set_neos_email(email):
        if solver is not OPT_DEFAULT and SolverFactory(solver).available():
            solver = SolverFactory(solver)
            return solver.solve(model, tee=True), 1
        elif cet == CET_ADDI:
            solver = SolverFactory("mosek")
            print("Estimating the additive model locally with mosek solver")
            return solver.solve(model, tee=True), 1
        elif cet == CET_MULT:
            raise ValueError(
                "Please specify the solver for optimizing multiplicative model locally.")
    else:
        if solver is OPT_DEFAULT and cet is CET_ADDI:
            solver = "mosek"
            print("Estimating the additive model remotely with mosek solver")
        elif solver is OPT_DEFAULT and cet == CET_MULT:
            solver = "knitro"
            print("Estimating the multiplicative model remotely with knitro solver")
        remote_solver = SolverManagerFactory('neos')
        return remote_solver.solve(model, tee=True, opt=solver), 1


def trans_list(li):
    if type(li) == list:
        return li
    return li.tolist()
