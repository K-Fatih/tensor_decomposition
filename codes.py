import time
import numpy as np

import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state

class TensDecom:
    def __init__(self, tensor):
        self.tensor = tensor.astype('float')
        self.size = tensor.size
        self.shape = tensor.shape
        self.mem_size = tensor.nbytes    
        self.decompositions = {}
    
    def svd(self):
        u, s, v = np.linalg.svd(self.tensor)
        return u,s,v
    def cp(self, rank=2):
        weights, factors = parafac(self.tensor,rank)
        return weights,factors
    def tucker(self, rank= []):
        weights, factors = tucker(self.tensor, rank)
        return weights, factors
    def tensor_train(self, rank=[]):
        weights, factors = matrix_product_state(self.tensor, rank)
        return weights, factors
    
    
       

a = np.random.randint(20, size = (12,12))


tens = TensDecom(a)

x,y = tens.tensor_train([1,2,1])

print(x,"\n\n",y)