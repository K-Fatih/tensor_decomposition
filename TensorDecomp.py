from timeit import default_timer as timer
import numpy as np


import matplotlib.pyplot as plt
from numpy.linalg import svd

from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state

decomp_list = ['svd', 'parafac', 'tucker', 'matrix_product_state']

class TensorDecomp:
    def __init__(self, tensor):        
        self.tensor = tensor
        self.size = tensor.size
        self.shape = tensor.shape
        self.mem_size = tensor.nbytes
        self.relative_error = None

    
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
            self.decomp_rel_error = np.linalg.norm(self.tensor - self.recons) / np.linalg.norm(self.tensor)
        elif self.decomp_type == 'tucker':
            from tensorly import tucker_tensor as tt
            self.recons = tt.tucker_to_tensor(self.decomposed)
            self.decomp_rel_error = np.linalg.norm(self.tensor - self.recons) / np.linalg.norm(self.tensor)
        elif self.decomp_type == 'parafac':
            from tensorly import cp_tensor as ct
            self.recons = ct.cp_to_tensor(self.decomposed)
            self.decomp_rel_error = np.linalg.norm(self.tensor - self.recons) / np.linalg.norm(self.tensor)
        elif self.decomp_type == 'matrix_product_state':
            from tensorly import tt_tensor as tt
            self.recons = tt.tt_to_tensor(self.decomposed)
            self.decomp_rel_error = np.linalg.norm(self.tensor - self.recons) / np.linalg.norm(self.tensor)

       
a = np.random.randint(20, size = (12,12))

tens = TensorDecomp(a.astype('float'))
tens.decompose(matrix_product_state, rank = [1,2,1])
tens.reconstruct()

print("Decomposition Type:\t", tens.decomp_type)
print("Decomposition Time:\t", tens.decomp_time)
print("Decomposition Relative Error:\t", tens.decomp_rel_error)


