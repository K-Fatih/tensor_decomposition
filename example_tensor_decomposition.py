# TODO preprocessing is missing
# TODO add examples how to use
# get timers out of the backend
# error with svd, rank is not considered
# error plots shouldn't be zero

import numpy as np
from scipy.sparse import bsr_matrix

def blk_diag(data):
    """Build a block diagonal sparse matrix from a 3d numpy array

    :param data: input array in R^{ngp,d1,d2}
    :type data: array

    :returns: sparse matrix in R^{ngp*d1, ngp*d2}

    Notes
    ----
        for the intended usage here, this function is faster than block_diag 'from scipy.sparse import block_diag'
        See Also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html
    """

    indptr = np.array(range(0, data.shape[0] + 1))
    indices = np.array(range(data.shape[0]))
    return bsr_matrix((data, indices, indptr))

norm = lambda x: np.linalg.norm(x)

# input data
ngp, d, N = 50, 3, 10

xi = np.random.rand(N)
stress = np.random.rand(ngp, d)
tangent = np.random.rand(ngp, d, d)
weights = np.ones(ngp) / ngp

tensor = np.random.rand(ngp, d, N)
mat = tensor.reshape(-1, N)

# same algebraic operations on the tensor or matrix representation
print((r11 := (tensor @ xi)).shape)
print((r12 := (mat @ xi).reshape(ngp, d)).shape)

print((r21 := (mat.T @ stress.flatten())).shape)
print((r22 := np.einsum('ilk,il->k', tensor, stress, optimize='optimal')).shape)

print((r31 := (mat.T @ blk_diag(tangent) @ mat)).shape)
print((r32 := np.einsum('ilk,ilm,imp->kp', tensor, tangent, tensor, optimize='optimal')).shape)

# error here should be zero
print(norm(r11 - r12), norm(r21 - r22), norm(r31 - r32))

# decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state', 'NMF', 'non_negative_parafac', 'clarkson_woodruff_transform']
from scipy.sparse.linalg import svds
# Compute the largest or smallest k singular values/vectors for a sparse matrix. The order of the singular values is not guaranteed.
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import matrix_product_state
from scipy.linalg import clarkson_woodruff_transform
from scipy.linalg._sketches import cwt_matrix
from sklearn.decomposition import NMF

rank = 2
# algebraic operations on the SVD representation
u, s, vt = svds(mat, rank)  # only to matrices
mat_approx_svd = u @ np.diag(s) @ vt
r13 = u @ (np.diag(s) @ (vt @ xi))
r23 = vt.T @ np.diag(s) @ ( u.T @ stress.flatten())
r33 = vt.T @ np.diag(s) @ ( u.T @ blk_diag(tangent) @ u) @ np.diag(s) @ vt

# sketching on matrices
sketch_n_rows = 30
sketch_mat = clarkson_woodruff_transform(mat,sketch_n_rows)
sketch_stress = clarkson_woodruff_transform(stress.flatten(),sketch_n_rows)
mat_stress = clarkson_woodruff_transform(np.hstack((stress.reshape(-1,1),mat)),sketch_n_rows)
# print(np.abs(norm(r13)-norm(sketch_mat@xi))/norm(r13))
## r24 = sketch_mat.T @ sketch_stress
r24 = mat_stress[:,1:].T @ mat_stress[:,0]
S = cwt_matrix(sketch_n_rows, blk_diag(tangent).shape[0], seed=None)
r34 = (mat.T @ S.T) @ (S @ blk_diag(tangent) @ S.T) @ (S @ mat)

weights, factors = parafac(tensor, rank)  # use with tensors [for matrices it'll match svd?]
# mat_approx_parafac = (factors[0]*weights) @ factors[1].T
tensor_approx_parafac = np.einsum('il,jl,kl->ijk',(factors[0]*weights),factors[1],factors[2])
r14 = np.einsum('il,jl,l->ij',(factors[0]*weights),factors[1],factors[2].T @ xi)

core, factors = tucker(tensor, (rank,rank,rank))
tensor_approx_tucker = np.einsum('lmn,il,jm,kn->ijk',core,factors[0],factors[1],factors[2])

factors = matrix_product_state(tensor, (1,rank,rank,1))
tensor_approx_tt = np.einsum('nil,ljm,mkn->ijk',factors[0],factors[1],factors[2])

