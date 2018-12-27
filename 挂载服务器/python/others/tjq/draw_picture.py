import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


name = ['quote000002', 'quote000006', 'quote000007', 'quote000011', 'quote000014', 'quote000024', 'quote000031', 'quote600533', 'quote600641', 'quote900911', 'quote900950']
colour = ['blue','green','cyan','magenta','yellow','red','black','maroon','salmon','C1','C0']
plt.figure(figsize=(30, 20))
for i in range(len(name)):

    data1 = pd.read_csv('./house/'+name[i]+'/lrb.csv',encoding='gbk',header=None)
    # print(data1.info())
    data1_x = np.array(data1.iloc[0,1:])
    data1_y = np.array(data1.iloc[1,1:].astype(int))
    # print(data1.iloc[1,1:])
    # print(type(data1_y[2]))
    # print(type(data1_y[2:4]))

    plt.subplot(6,2, i+1)
    plt.plot(data1_x,data1_y,color=colour[i])

    plt.title(name[i])
    plt.tight_layout(2)
plt.show()



