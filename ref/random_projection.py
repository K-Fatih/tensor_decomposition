import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
y = np.random.rand(10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)

from sklearn.random_projection import johnson_lindenstrauss_min_dim
print(johnson_lindenstrauss_min_dim(n_samples=1e4, eps=0.15))

import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
y_new = transformer.fit_transform(y.reshape(1, -1))
print(X_new.shape)

print(np.linalg.norm(X @ y))
print(np.linalg.norm(X @ y))
