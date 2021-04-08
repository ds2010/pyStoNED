# import dependencies
import re
import os
from pyomo.opt import SolverFactory, SolverManagerFactory
from ..constant import CET_ADDI, CET_MULT, OPT_LOCAL, OPT_DEFAULT
__email_re = re.compile(r'([^@]+@[^@]+\.[a-zA-Z0-9]+)$')


def set_neos_email(address):
    """pass email address to NEOS server 

    Args:
        address (String): your own vaild email address.
    """
    if address == OPT_LOCAL:
        print("Optimizing locally.")
        return False
    if not __email_re.match(address):
        # TODO: Replace with log system
        print("Invalid email address.\n")
        return False
    os.environ['NEOS_EMAIL'] = address
    return True


def optimize_model(model, email, cet, solver=OPT_DEFAULT):
    if solver is not OPT_DEFAULT and SolverFactory(solver).available():
        solver = SolverFactory(solver)
        return solver.solve(model, tee=True), 1

    if not set_neos_email(email):
        if cet == CET_ADDI:
            solver = SolverFactory("mosek")
            print("Estimating the additive model locally with mosek solver")
            return solver.solve(model, tee=True), 1
        elif cet == CET_MULT:
            # TODO(warning handling): Use log system instead of print()
            print(
                "Estimating the multiplicative model will be available in near future."
            )
            return False, 0
    else:
        if cet == CET_ADDI:
            opt = "mosek"
            print("Estimating the additive model remotely with mosek solver")
        elif cet == CET_MULT:
            opt = "knitro"
            print("Estimating the multiplicative model remotely with knitro solver")
        solver = SolverManagerFactory('neos')
        return solver.solve(model, tee=True, opt=opt), 1
