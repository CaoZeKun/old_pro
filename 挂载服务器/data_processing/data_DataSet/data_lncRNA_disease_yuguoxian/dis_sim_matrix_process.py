import numpy as np

# a = [[0,1,2,3],
#      [0,0,4,5],
#      [0,0,0,6],
#      [0,0,0,0]]
#
#
# for i in range(len(a)):
#     j = i + 1
#     while j < len(a[0]):
#         a[j][i] = a[i][j]
#         j += 1

dis_sim = np.loadtxt('./data_create/diseases_sim.txt')

for i in range(len(dis_sim)):
    j = i + 1
    while j < len(dis_sim[0]):
        dis_sim[j][i] = dis_sim[i][j]
        j += 1
for i in range(len(dis_sim)):
    dis_sim[i][i] = 1.0
np.savetxt('./data_create/dis_sim_matrix_process.txt',dis_sim)


