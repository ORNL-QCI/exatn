import numpy as np
from openfermion import *
import openfermion.ops as ops
import openfermion.hamiltonians as hams
import openfermion.transforms as trans
import openfermion.linalg as linalg

hubb = hams.fermi_hubbard(2,2,1,1)
hubb_jw = trans.jordan_wigner(hubb)
print(hubb_jw)

hubb_matrix = linalg.get_sparse_operator(hubb).A
ground_state = linalg.get_ground_state(linalg.get_sparse_operator(hubb))
print(ground_state)

spectrum = linalg.eigenspectrum(hubb)
print(spectrum)
