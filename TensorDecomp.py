import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import svd
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state
from timeit import default_timer as timer
from sklearn.decomposition import NMF

decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state', 'NMF']

class TensorDecomp:
    def __init__(self, tensor):        
        self.tensor = tensor
        self.size = tensor.size
        self.shape = tensor.shape
        self.memSize = tensor.nbytes
        self.decMemSize = 0

    
    def decompose(self, func, *args, **kwargs):        

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
        

    def reconstruct(self):

        if self.decomp_type == 'svd':
            self.recons = self.decomposed[0] @ (np.diag(self.decomposed[1])@self.decomposed[2])            

        elif self.decomp_type == 'tucker':
            from tensorly import tucker_tensor as tt
            self.recons = tt.tucker_to_tensor(self.decomposed)            

        elif self.decomp_type == 'parafac':
            from tensorly import cp_tensor as ct
            self.recons = ct.cp_to_tensor(self.decomposed)            

        elif self.decomp_type == 'matrix_product_state':
            from tensorly import tt_tensor as tt
            self.recons = tt.tt_to_tensor(self.decomposed)
        elif self.decomp_type == 'NMF':
            self.recons = self.nmf_obj.inverse_transform(self.decomposed[0])

 

        for array in self.decomposed:
            if isinstance(array,(np.ndarray)):        
                self.decMemSize += array.nbytes
            for array in self.decomposed[1]:
                if isinstance(array,(np.ndarray)):
                    self.decMemSize += array.nbytes

    # def vecMult(self, vec):
    #     tenVec = self.tensor @ vec
    #     decVec = self.recons @ vec
    #     self.vecMultErr = np.linalg.norm(tenVec - decVec) / np.linalg.norm(tenVec)

    def error(self,func, x, y):

        return func(x-y) / func(x)


a = np.random.randint(20, size = (12,12))
vec = np.random.randint(5, size = a.shape[1])


tens = TensorDecomp(a.astype('float'))
tens.decompose(NMF, rank = 10)
tens.reconstruct()




print(f"Decomposition Type:\t\t\t {tens.decomp_type}")
print(f"Size in memory before decomposition:{tens.memSize: 9d}")
print(f"Decomposition Time: {tens.decomp_time:39.16f}")
print(f"Size in memory after decomposition:\t {tens.decMemSize}")
print(f"Decomposition Relative Error:{tens.error(np.linalg.norm, tens.tensor, tens.recons):30.16f}", )
# print(f"Vector Multiplication Error:{tens.vecMultErr:31.16f}" )