import DLACNNLDA as DL

import numpy as np



A, Sd, Lm, Md = DL.read_data_flies()

print(np.shape(A))
print(np.sum(A,axis=0))

print(np.where(np.sum(A,axis=0)>80))



