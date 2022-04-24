
# from os.path import dirname, basename, isfile, join
# import glob

__all__ = ['local_buck_NEAT', 'common', 'funciones']

# ---------------------------------------------------------------------------------------------------
# Alternativa para hacer el listado automático (de momento lo dejo, ..pero para cuando crezca la lista)
# A utilizar en todos los subpackages también

# modules = glob.glob(join(dirname(__file__), "*.py"))       # Lista con las direcciones

# __all__ = [ basename(f)[:-3] for f in modules             # List comprehension y comprueba que
#            if isfile(f) and not f.endswith('__init__.py')]# sean archivos y que no sea el __ini__
# ----------------------------------------------------------------------------------------------------

from . import *
