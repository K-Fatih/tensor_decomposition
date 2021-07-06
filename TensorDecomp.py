from math import inf
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import svd
from scipy.linalg import norm
from scipy.linalg import clarkson_woodruff_transform
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state
from timeit import default_timer as timer
from sklearn.decomposition import NMF
from tensorly.contrib.sparse.decomposition import tucker as sparse_tucker


decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state', 'NMF','sparse_tucker', 'clarkson_woodruff_transform']


class TensorDecomp:
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
    memChange   : float
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

        self.tensor = tensor      
        self.decMemSize = 0
        self.memSize = tensor.nbytes
        
    def decompose(self, func, *args, **kwargs):
        """
        Decomposes the tensor with the func argument decomposition type.
        Assigns the objects after decomposition to self.recons.
        Computes the decomposition time and assigns to self.decomp_time.
        Assigns the func argument as the decomposition type to self.decomp_type.

        Parameters
        ----------
        self    : object of class TensorDecomp type.
        func    : the decomposition type. e.g.: parafac, NMF etc.
        *args   : passed arguments.
        **kwargs: passed keyword arguments. e.g.: rank, sketch_size etc.
            
        Returns
        -------
        None            
        
        """

        if func.__name__ not in decomp_list:
            print(f'Error! Given decomposition --> {func.__name__}')
            return

        elif func.__name__ == 'svd':
            ts = timer()
            self.recons = func(self.tensor)
            te = timer()
            self.decomp_time = te-ts
            self.decomp_type = func.__name__

        elif func.__name__ == 'NMF':
            self.nmf_obj = NMF()
            ts = timer()
            self.recons = []
            self.recons.append(self.nmf_obj.fit_transform(self.tensor) )
            self.recons.append(self.nmf_obj.components_)
            te = timer()
            self.decomp_time = te-ts
            self.decomp_type = func.__name__

        elif args:
            ts = timer()
            self.recons = func(self.tensor, args[0])
            te = timer()
            self.decomp_type = func.__name__
            self.decomp_time = te-ts

        else:
            ts = timer()
            self.recons = func(self.tensor, **kwargs)
            te = timer()
            self.decomp_type = func.__name__
            self.decomp_time = te-ts

        for array in self.recons:
            if isinstance(array,(np.ndarray)):
                self.decMemSize += array.nbytes
            for array in self.recons[1]:
                if isinstance(array,(np.ndarray)):
                    self.decMemSize += array.nbytes

        # the tensor size change in memory

        self.memChange = (self.memSize - self.decMemSize) / self.memSize

    def reconstruct(self):
        """
        Reconstructs the recons TensorDecomp object.
        Assigns the reconstructed tensor to self.recons attribute.

        Parameters
        ----------
        self : object of class TensorDecomp type.
            
        Returns
        -------
        None            
        
        """

        if self.decomp_type == 'svd':
            self.recons = self.recons[0] @ (np.diag(self.recons[1])@self.recons[2])

        elif self.decomp_type == 'NMF':
            self.recons = self.nmf_obj.inverse_transform(self.recons[0])

        elif self.decomp_type == 'tucker':
            from tensorly import tucker_tensor as tt
            self.recons = tt.tucker_to_tensor(self.recons)

        elif self.decomp_type == 'parafac':
            from tensorly import cp_tensor as ct
            self.recons = ct.cp_to_tensor(self.recons)

        elif self.decomp_type == 'matrix_product_state':
            from tensorly import tt_tensor as tt
            self.recons = tt.tt_to_tensor(self.recons)

        elif self.decomp_type == 'clarkson_woodruff_transform':
            self.recons = self.recons

    def error(self,func, x, y):
        """
        Computes the error between the original and reconstructed tensor with a given error function.

        Parameters
        ----------
        func    : function object for error calculation. Example: np.linalg.norm
        x       : the original tensor
        y       : the reconstructed tensor
            
        Returns
        -------
        float
            the error between the original and the reconstructed tensor.            
        
        """

        return (func(x-y)) / func(x)

def errList(tensor, decompMet, vectorL, vectorR, MatrixL, MatrixR, uNorm = [], rank = [2,3]):
    """A function to calculate the Frobenius, L1, L_inf, and Spectral-Norm of
    the tensor decomposition, Tensor @ Vector, Vector @ Tensor, Tensor @ Matrix0, Matrix @ Tensor, 
    Vector.T @ Tensor @ Vector, Matrix.T @ Tensor @ Matrix.    

    Parameters
    ----------
    tensor (TensorDecomp)   : TensorDecomp object.
    decompMet (Function)    : The decomposition method to decompose the TensorDecomp object.
    vectorL (numpy.ndarray) : A vector of size tensor.tensor.shape([1])
    vectorR (numpy.ndarray) : A vector of size tensor.tensor.shape([-1])
    MatrixL (numpy.ndarray) : A matrix of size tensor.tensor.shape([:-1])
    MatrixR (numpy.ndarray) : A matrix of size tensor.tensor.shape([-1:-3:-1])
    uNorm                   : A list of user norm functions. (optional)
    rank                    : Optional argument to decompose with the relatead rank. (optional)

    Returns
    -------

    A list of 7 lists each contains the Frobenius, L1, L_inf, Spectral Norm of errors of Tensor operations.

    """
    tensor.decompose(decompMet, rank)
    tensor.reconstruct()

    err_fro  = lambda  x, y : norm(x-y, ord='fro') / norm(x, ord='fro') if (x.ndim == 2) else norm(x-y, ord=None) / norm(x, ord=None)
    err_L1   = lambda  x, y : norm(x-y, ord=1) / norm(x, ord=1) if (x.ndim == 2) else norm(x-y, ord=None) / norm(x, ord=None)
    err_Linf = lambda  x, y : norm(x-y, ord=inf) / norm(x, ord=inf) if (x.ndim == 2) else norm(x-y, ord=None) / norm(x, ord=None)
    err_Spec = lambda  x, y : norm(x-y, ord=2) / norm(x, ord=2) if (x.ndim == 2) else norm(x-y, ord=None) / norm(x, ord=None)
    
    normL = [err_fro, err_L1, err_Linf, err_Spec]
    normL.append(*uNorm) if len(uNorm)!=0 else None


    tensVec = [norm(tensor.tensor@vectorR, tensor.recons@vectorR) for norm in normL]
    vecTens = [norm(vectorL@tensor.tensor, vectorL@tensor.recons) for norm in normL]
    matLTens = [norm(MatrixL@tensor.tensor, MatrixL@tensor.recons) for norm in normL]
    tensMatR = [norm(tensor.tensor@MatrixR, tensor.recons@MatrixR) for norm in normL]
    vectTenvec = [norm(vectorL@tensor.tensor@vectorR, vectorL@tensor.recons@vectorR) for norm in normL]
    matTTenMat = [norm(MatrixL@tensor.tensor@MatrixR, MatrixL@tensor.recons@MatrixR) for norm in normL]
    decErr = [norm(tensor.tensor, tensor.recons) for norm in normL]

    return [decErr, tensVec, vecTens, matLTens, tensMatR, vectTenvec, matTTenMat]

