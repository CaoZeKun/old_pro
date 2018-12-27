import pandas as pd
import numpy as np




def get_index(lncrna):
    index = []
    for i in range(len(lncrna)):
        if lncrna[i] == 1 :
            index.append(i)

    return index


def get_sim(l1_disease,l2_diseases,disease_sim):
    dis_sims = []
    for i in range(len(l2_diseases)):
        dis_sims.append([disease_sim[l1_disease,l2_diseases[i]]])
    return np.max(dis_sims)

def get_sim_all(l1_diseases,l2_diseases,disease_sim):
    diss_sims1 = []
    for i in range(len(l1_diseases)):
        dis_sim_temp = get_sim(l1_diseases[i],l2_diseases,disease_sim)
        diss_sims1.append(dis_sim_temp)
    l1_dis = np.sum(diss_sims1)

    diss_sims2 = []
    for i in range(len(l2_diseases)):
        dis_sim_temp = get_sim(l2_diseases[i], l1_diseases,disease_sim)
        diss_sims2.append(dis_sim_temp)
    l2_dis = np.sum(diss_sims2)

    l_dis_sim = np.sum((l1_dis,l2_dis))
    return l_dis_sim


def get_sim_value(lncrna1,lncrna2,disease_sim):  # ln1 = A[0]

    lnc1_dis_index = get_index(lncrna1)
    lnc2_dis_index = get_index(lncrna2)


    lnc1_dis_len = len(lnc1_dis_index)
    lnc2_dis_len = len(lnc2_dis_index)
    length_l1_l2 = lnc1_dis_len + lnc2_dis_len
    if lnc1_dis_len==0 or lnc2_dis_len==0:
        sim = 0
    else:
        l_dis_sim = get_sim_all(lnc1_dis_index,lnc2_dis_index,disease_sim)
        sim = l_dis_sim / (length_l1_l2)
    return sim



def get_all_sim(lnc_dis,dis_sim):
    lnc_length = len(lnc_dis)
    lnc_sim = np.zeros((lnc_length,lnc_length))
    print(np.shape(lnc_sim))

    for i in range(lnc_length):
        for j in range(lnc_length):
            sim = get_sim_value(lnc_dis[i],lnc_dis[j],dis_sim)
            lnc_sim[i,j] = sim
    for i in range(lnc_length):
        lnc_sim[i, i] = 1
    np.savetxt('./data_create/lnc_sim.txt',lnc_sim)

# test
# l = [[0,1,1,1],
#      [0,0,0,0],
#      [0,1,0,1],
#      [0,0,0,0]]
# d = [[0.1,0.1,0.2,0.3],
#      [0.1,0.1,0.4,0.5],
#      [0.1,0.2,0.3,0.6],
#      [0.2,0.3,0.2,0.3]]
#
# def get_all_sim1(lnc_dis,dis_sim):
#     lnc_length = len(lnc_dis)
#     lnc_sim = np.zeros((lnc_length,lnc_length))
#     print(np.shape(lnc_sim))
#
#     for i in range(lnc_length):
#         for j in range(lnc_length):
#             sim = get_sim_value(lnc_dis[i],lnc_dis[j],dis_sim)
#             lnc_sim[i,j] = sim
#     print(lnc_sim)
    #np.savetxt('./data_create/lnc_sim.txt',lnc_sim)

# l = np.array(l)
# d = np.array(d)
# get_all_sim1(l,d)


dis_sim = np.loadtxt('./data_create/dis_sim_matrix_process.txt', encoding='gbk')
# print(len(dis_sim))
A_lnc_dis = np.loadtxt('./data_create/lnc_dis_association.txt', encoding='gbk')

#get_all_sim(A_lnc_dis,dis_sim)

lnc_sim = np.loadtxt('./data_create/lnc_sim.txt',encoding='gbk')  # 0 24839  24780( eye 1 )
count0 = 0
count1 = 0
# count = 0
# for i in range(len(A_lnc_dis)):
#     for j in range(len(A_lnc_dis[0])):
#         if A_lnc_dis[i,j] == 1 :
#             count +=1
# print(count) # 2687


for i in range(len(lnc_sim)):
    #for j in range(len(lnc_sim)):
    if np.sum(A_lnc_dis[i]) == 0:
    #if np.sum(lnc_sim[i]) == 1:
        # if lnc_sim[i,j] == 0:
        count0 += 1
        print(i)

    # if np.sum(lnc_sim[i]) == 1:
    #     count1 += 1
    #     print(i)

        #     count += 1
print(count0)
print(count1)
