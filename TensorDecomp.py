from math import inf
import numpy as np
import timeit

from scipy.sparse.linalg import svds
from scipy.linalg import norm
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import matrix_product_state
from scipy.linalg import clarkson_woodruff_transform
from sklearn.decomposition import NMF
from scipy.sparse import bsr_matrix


decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state', 'NMF', 'clarkson_woodruff_transform', 'non_negative_parafac']

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
class TensorDecomp():
    """
    A class to represent a tensor object with decomposition, reconstruction, error methods.

    
    Attributes
    ----------
    tensor      : numpy.ndarray
        The tensor that is given to the class.
    memSize     : int
        The size of the tensor in the memory before decomposition.
    decMemSize  : int
        The size of the tensor in the memory after decomposition.
    decomp_time : float
        The time elapsed to decompose the tensor.
    decomp_type : str
        The __name__ of the provided func argument.
    memSaving   : float
        The relative change of memory requirement of the tensor after decomposition.


    Methods
    -------
    decompose(func, *args, **kwargs):
        Decomposes the given tensor with the 'func' decomposition and computes the size in memory after decomposition.
    
    reconstruct(self):
        Reconstructs the recons tensor.
    
    error(func, x, y):
        Calculates the error between x and y with the given 'func' error handle.

    """
    def __init__(self, tensor):
        """Creates the TensorDecomp object with given numpy.ndarray object.

        Args:
            tensor (numpy.ndarray): The tensor to be decomposed.
        """

        self.tensor = tensor      
        self.decMemSize = 0
        self.memSize = tensor.nbytes
        
    def decompose(self, func, *args, **kwargs):
        """
        Decomposes the tensor with the func argument decomposition type.
        Assigns the objects after decomposition to self.recons.
        Computes the decomposition time and assigns to self.decomp_time.
        Assigns the func argument as the decomposition type to self.decomp_type.

        Args:
        ----------
        func    : the decomposition type. e.g.: parafac, NMF etc.
        *args   : passed arguments.
        **kwargs: passed keyword arguments. e.g.: rank, sketch_size etc.
            
        Returns
        -------
        None            
        
        """
        n = 50

        if func.__name__ not in decomp_list:
            print(f'Error! Given decomposition --> {func.__name__}')
            return

        elif func.__name__ == 'svd':
            self.decomposed = func(self.tensor)
            self.decomp_time = timeit.timeit( lambda: func(self.tensor, **kwargs), number = n)/n
            self.decomp_type = func.__name__

        elif func.__name__ == 'NMF':
            try:
                self.nmf_obj = NMF(**kwargs)
                self.decomposed = []
                self.decomposed.append(self.nmf_obj.fit_transform(self.tensor) )
                self.decomposed.append(self.nmf_obj.components_)
                self.decomp_time = timeit.timeit(lambda: func(self.tensor, **kwargs), number = n)/n
                self.decomp_type = func.__name__
            except:
                raise

        elif args:            
            self.decomposed = func(self.tensor, args[0])
            self.decomp_type = func.__name__
            # self.decomp_time = timeit.timeit(stmt = 'func(self.tensor, args[0])', setup='from __main__ import func, tensor, args', number = n)/n
        else:
            self.decomposed = func(self.tensor, **kwargs)
            self.decomp_type = func.__name__
            # self.decomp_time = timeit.timeit(stmt = 'func(self.tensor, args[0])', globals=globals(), number = n)/n
        
        for array in self.decomposed:            
            if isinstance(array,(np.ndarray)):
                self.decMemSize += array.nbytes
            for array in self.decomposed[1]:
                if isinstance(array,(np.ndarray)):
                    self.decMemSize += array.nbytes

        # the tensor size change in memory
        self.memSaving = (self.memSize - self.decMemSize) / self.memSize

    def reconstruct(self):
        """
        Reconstructs the recons TensorDecomp object.
        Assigns the reconstructed tensor to self.recons attribute.

        Args:
        ----------
        None

        Returns
        -------
        None            
        
        """

        if self.decomp_type == 'svd':
            if self.tensor.ndim < 3:
                self.recons = self.decomposed[0][...,:self.tensor.shape[-1]]*self.decomposed[1]@self.decomposed[2]
            else:
                self.recons = np.matmul(self.decomposed[0][...,:self.tensor.shape[-1]], self.decomposed[1][..., None] * self.decomposed[2])

        elif self.decomp_type == 'NMF':
            self.recons = self.nmf_obj.inverse_transform(self.decomposed[0])

        elif self.decomp_type == 'tucker' :
            from tensorly import tucker_tensor as tt
            self.recons = tt.tucker_to_tensor(self.decomposed)

        elif self.decomp_type == 'parafac' or self.decomp_type == 'non_negative_parafac':
            from tensorly import cp_tensor as ct
            self.recons = ct.cp_to_tensor(self.decomposed)

        elif self.decomp_type == 'matrix_product_state':
            from tensorly import tt_tensor as tt
            self.recons = tt.tt_to_tensor(self.decomposed)

        elif self.decomp_type == 'clarkson_woodruff_transform':
            self.recons = self.decomposed


        self.decError = (norm(self.tensor-self.recons)) / norm(self.tensor)

def errList(tensor, decompMet, vectorL, MatrixR, MatrixL, normL, rank = None, **kwargs):
    """A function to calculate the norm of the following:

    tensor decomposition, 
    Tensor @ Vector,  
    Matrix @ Tensor,
    Matrix.T @ Tensor @ Matrix operations error, 

    in this order.    

    Args:
    ----------
    tensor (TensorDecomp)   : TensorDecomp object.
    decompMet (Function)    : The decomposition method to decompose the TensorDecomp object.
    vectorL (numpy.ndarray) : A vector of size tensor.tensor.shape([1])
    vectorR (numpy.ndarray) : A vector of size tensor.tensor.shape([-1])
    MatrixL (numpy.ndarray) : A matrix of size tensor.tensor.shape([:-1])
    MatrixR (numpy.ndarray) : A matrix of size tensor.tensor.shape([-1:-3:-1])
    normL (list)            : A list of norm functions to calculate the norm of the errors.
    rank (int or list)      : To decompose with the relatead rank. (optional)

    Returns
    -------

    A list of 7 lists each contains the norm of the errors of the above tensor operations.

    """
    if tensor.tensor.ndim != 2 and decompMet in [NMF, clarkson_woodruff_transform] :
        return (f"It is not possible to decompose {tensor.tensor.ndim} dimension with the {decompMet.__name__} method!")           
    
    if decompMet in [NMF,clarkson_woodruff_transform]:
        tensor.decompose(decompMet, **kwargs)
        tensor.reconstruct()        
    else:
        tensor.decompose(decompMet, rank)
        tensor.reconstruct()

    decErr = None

    tensVec = [norm(tensor.tensor@vectorR, tensor.recons@vectorR) for norm in normL]
    vecTens = [norm(vectorL@tensor.tensor, vectorL@tensor.recons) for norm in normL]
    matLTens = [norm(MatrixL@tensor.tensor, MatrixL@tensor.recons) for norm in normL]
    tensMatR = [norm(tensor.tensor@MatrixR, tensor.recons@MatrixR) for norm in normL]
    vectTensvec = [norm(vectorL@tensor.tensor@vectorR, vectorL@tensor.recons@vectorR) for norm in normL]
    matLTensMatR = [norm(MatrixL@tensor.tensor@MatrixR, MatrixL@tensor.recons@MatrixR) for norm in normL]

    if decompMet == clarkson_woodruff_transform:        
        return [decErr, tensVec, vecTens, tensMatR, matLTens, vectTensvec, matLTensMatR]
    else:
        decErr = [norm(tensor.tensor, tensor.recons) for norm in normL]
        return [decErr, tensVec, vecTens, tensMatR, matLTens, vectTensvec, matLTensMatR]

def tensOpTim(operList):
    """Performs tensor@vectorR, vectorL@tensor, tensor@MatrixR, MatrixL@tensor, vecL@tensor@vecR, matL@tensor@matR
    operations and times the each operation returns them in a list in the explained order

    Args:
        operList    : List of tensor operations in lambda form

    Returns:
        list        : List of tensor operation times
    """
    timing = []
    n = 50
    for operation in operList:
        opTime = timeit.timeit(operation, number=n)/n
        timing.append(opTime)
    # opTime = timeit.timeit(lambda:tensor.tensor@vecR, number = n)/n
    # timing.append(opTime)
    # opTime = timeit.timeit(lambda:vecL@tensor.tensor, number = n)/n
    # timing.append(opTime)
    # opTime = timeit.timeit(lambda:tensor.tensor@matR, number = n)/n
    # timing.append(opTime)
    # opTime = timeit.timeit(lambda:matL@tensor.tensor, number = n)/n
    # timing.append(opTime)
    # opTime = timeit.timeit(lambda:vecL@tensor.tensor@vecR, number = n)/n
    # timing.append(opTime)
    # opTime = timeit.timeit(lambda:matL@tensor.tensor@matR, number = n)/n
    # timing.append(opTime)
    return timing

### NOT Implemented ###
# def decTensOpTim(tensor, decompMet, vectorR, vectorL, MatrixR, MatrixL, rank = None, **kwargs):
#     timing = []
#     if tensor.tensor.ndim != 2 and decompMet in [NMF, clarkson_woodruff_transform] :
#         return (f"It is not possible to decompose {tensor.tensor.ndim = } with the {decompMet.__name__} method!")           
    
#     if decompMet in [NMF,clarkson_woodruff_transform]:
#         tensor.decompose(decompMet, **kwargs)
#     else:
#         tensor.decompose(decompMet, rank)

    
#     operList = []

#     for operation in operList:
#         t1 = timer()
#         eval(operation)
#         t2 = timer()
#         timing.append((t2-t1))

#     return timing