import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


from numpy.linalg import svd
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state
from timeit import default_timer as timer
from sklearn.decomposition import NMF
from tensorly.contrib.sparse.decomposition import tucker as tucker_s

decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state', 'NMF','stucker']

class TensorDecomp:
    """
    A class to represent a tensor object with decomposition and reconstruction methods.

    
    Attributes
    ----------
    tensor : numpy.ndarray
        The tensor that is given to the class.
    memSize : int
        The size of the tensor in the memory before decomposition.
    decMemSize : int
        The size of the tensor in the memory after decomposition.
    decomp_time : float
        The time elapsed to decompose the tensor.
    decomp_type : str
        The __name__ of the provided func argument.


    Methods
    -------
    decompose(func, *args, **kwargs):
        Decomposes the given tensor with the 'func' decomposition and computes the size in memory after decomposition.
    
    reconstruct(self):
        Reconstructs the decomposed tensor.
    
    error(func, x, y):
        Calculates the error between x and y with the given 'func' error handle.

    """
    def __init__(self, tensor):

        self.tensor = tensor      
        self.decMemSize = 0

        if sparse.issparse(tensor):
            print("Sparse!!!!")
            self.memSize = tensor.data.nbytes + tensor.row.nbytes + tensor.col.nbytes
        else:
            self.memSize = tensor.nbytes
        
    def decompose(self, func, *args, **kwargs):
        """
        Decomposes the tensor with the func argument decomposition type.
        Assigns the objects after decomposition to self.decomposed.
        Computes the decomposition time and assigns to self.decomp_time.
        Assigns the func argument as the decomposition type to self.decomp_type.

        Parameters
        ----------
        self : object of class TensorDecomp type.
            
        Returns
        -------
        None            
        
        """

        if func.__name__ not in decomp_list:
            print(f'Error! Given decomposition --> {func.__name__}')
            return

        elif func.__name__ == 'svd':
            ts = timer()
            self.decomposed = func(self.tensor)
            te = timer()
            self.decomp_time = te-ts
            self.decomp_type = func.__name__

        elif func.__name__ == 'NMF':
            self.nmf_obj = NMF()
            ts = timer()
            self.decomposed = []
            self.decomposed.append(self.nmf_obj.fit_transform(self.tensor) )
            self.decomposed.append(self.nmf_obj.components_)
            te = timer()
            self.decomp_time = te-ts
            self.decomp_type = func.__name__

        elif args:
            ts = timer()
            self.decomposed = func(self.tensor, args[0])
            te = timer()
            self.decomp_type = func.__name__
            self.decomp_time = te-ts

        elif 'rank' in kwargs:
            ts = timer()
            self.decomposed = func(self.tensor, kwargs['rank'])
            te = timer()
            self.decomp_type = func.__name__
            self.decomp_time = te-ts

        else:
            ts = timer()
            self.decomposed = func(self.tensor)
            te = timer()
            self.decomp_type = func.__name__
            self.decomp_time = te-ts

        for array in self.decomposed:
            if isinstance(array,(np.ndarray)):
                self.decMemSize += array.nbytes
            for array in self.decomposed[1]:
                if isinstance(array,(np.ndarray)):
                    self.decMemSize += array.nbytes

    def reconstruct(self):
        """
        Reconstructs the decomposed TensorDecomp object.
        Assigns the reconstructed tensor to self.recons attribute.

        Parameters
        ----------
        self : object of class TensorDecomp type.
            
        Returns
        -------
        None            
        
        """

        if self.decomp_type == 'svd':
            self.recons = self.decomposed[0] @ (np.diag(self.decomposed[1])@self.decomposed[2])

        elif self.decomp_type == 'NMF':
            self.recons = self.nmf_obj.inverse_transform(self.decomposed[0])

        elif self.decomp_type == 'tucker':
            from tensorly import tucker_tensor as tt
            self.recons = tt.tucker_to_tensor(self.decomposed)

        elif self.decomp_type == 'parafac':
            from tensorly import cp_tensor as ct
            self.recons = ct.cp_to_tensor(self.decomposed)

        elif self.decomp_type == 'matrix_product_state':
            from tensorly import tt_tensor as tt
            self.recons = tt.tt_to_tensor(self.decomposed)

    def error(self,func, x, y):
        """
        Computes the error between the original and reconstructed tensor with a given error function.

        Parameters
        ----------
        func    : function object
        x       : the original tensor
        y       : the reconstructed tensor
            
        Returns
        -------
        float
            the error between the original and the reconstructed tensor.            
        
        """

        return func(x-y) / func(x)

