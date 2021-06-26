#%% https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.clarkson_woodruff_transform.html
# Given an input_matrix A of size (n, d), compute a matrix A' of size (sketch_size, d) so that

import numpy as np
from scipy import linalg
from scipy import sparse
n_rows, n_columns, density, sketch_n_rows = 15000, 100, 0.01, 200
A = sparse.rand(n_rows, n_columns, density=density, format='csc')
B = sparse.rand(n_rows, n_columns, density=density, format='csr')
C = sparse.rand(n_rows, n_columns, density=density, format='coo')
D = np.random.randn(n_rows, n_columns)
SA = linalg.clarkson_woodruff_transform(A, sketch_n_rows)  # fastest
SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows)  # fast
SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows)  # slower
SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows)  # slowest

#%%
n_rows, n_columns, sketch_n_rows = 15000, 100, 200
A = np.random.randn(n_rows, n_columns)
sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows)
sketch.shape

norm_A = np.linalg.norm(A)
norm_sketch = np.linalg.norm(sketch)
print((norm_A - norm_sketch)/norm_A)
#%%
n_rows, n_columns, sketch_n_rows = 15000, 100, 200
A = np.random.randn(n_rows, n_columns)
b = np.random.randn(n_rows)
x = np.linalg.lstsq(A, b, rcond=None)
Ab = np.hstack((A, b.reshape(-1, 1)))
SAb = linalg.clarkson_woodruff_transform(Ab, sketch_n_rows)
SA, Sb = SAb[:, :-1], SAb[:, -1]
x_sketched = np.linalg.lstsq(SA, Sb, rcond=None)
