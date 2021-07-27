# import dependencies
from re import compile
from os import environ
from pyomo.opt import SolverFactory, SolverManagerFactory
from ..constant import CET_ADDI, CET_MULT, OPT_LOCAL, OPT_DEFAULT, RTS_CRS
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


def assert_optimized(optimization_status):
    if optimization_status == 0:
        raise Exception(
            "Model isn't optimized. Use optimize() method to estimate the model.")


def assert_contextual_variable(z):
    if type(z) == type(None):
        raise Exception(
            "Estimated coefficient (lambda) cannot be retrieved due to no contextual variable (z variable) included in the model.")


def assert_undesirable_output(b):
    if type(b) == type(None):
        raise Exception(
            "Estimated coefficient (delta) cannot be retrieved due to no undesirable output (b variable) included in the model.")


def assert_various_return_to_scale(rts):
    if rts == RTS_CRS:
        raise Exception(
            "Estimated intercept (alpha) cannot be retrieved due to the constant returns-to-scale assumption.")
