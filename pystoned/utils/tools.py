# import dependencies
import re
import os
from ..constant import OPT_LOCAL
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
