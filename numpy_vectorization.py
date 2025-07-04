# Outline
#   1.1 Goals
#   1.2 Useful References
# 2 Python and NumPy
# 3 Vectors
#   3.1 Abstract
#   3.2 NumPy Arrays
#   3.3 Vector Creation
#   3.4 Operations on Vectors
# 4 Matrices
#   4.1 Abstract
#   4.2 NumPy Arrays
#   4.3 Matrix Creation
#   4.4 Operations on Matrices

import numpy as np  # it is an unofficial standard to use np for numpy
import time

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(
    f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}"
)
