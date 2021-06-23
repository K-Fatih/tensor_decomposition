from timeit import default_timer as timer
import numpy as np

import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import matrix_product_state

def timeit(method):

    def timed(*args, **kw):
        ts = timer()
        result = method(*args, **kw)
        te = timer()

        #print ('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        return te-ts, result

    return timed

class TensorDecomp:
    def __init__(self, tensor):
        self.tensor = tensor
        self.size = tensor.size
        self.shape = tensor.shape
        self.mem_size = tensor.nbytes    
        self.decomposition_type = ''   

    @timeit
    def svd(self: 'TensorDecomp') -> 'list':
        """

        """       
        u, s, v = np.linalg.svd(self.tensor)
        return {f"SVD: Rank:": [u,s,v]}

    @timeit
    def cp(self, rank=2):
        """

        """
        weights, factors = parafac(self.tensor,rank)
        return {f"CP: Rank: {rank}": [weights, factors]}

    @timeit
    def tucker(self, rank= []):
        """
        
        """
        weights, factors = tucker(self.tensor, rank)
        return {f"Tucker: Rank: {rank}": [weights, factors]}

    @timeit
    def tensor_train(self, rank=[]):
        """
        
        """ 
        weights, factors = matrix_product_state(self.tensor, rank)
        return {f"TensorTrain with Rank: {rank}": [weights, factors]}
    
    
       

a = np.random.randint(20, size = (40,40))


tens = TensorDecomp(a.astype('float'))

y = tens.tensor_train([1,2,1])

print(y)