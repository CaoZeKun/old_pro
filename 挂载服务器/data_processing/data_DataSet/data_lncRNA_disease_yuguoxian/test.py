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



# def get_all_sim(lnc_dis,dis_sim):
#     lnc_length = len(lnc_dis)
#     lnc_sim = np.zeros((lnc_length,lnc_length))
#     print(np.shape(lnc_sim))
#
#     for i in range(lnc_length):
#         for j in range(lnc_length):
#             sim = get_sim_value(lnc_dis[i],lnc_dis[j],dis_sim)
#             lnc_sim[i,j] = sim
#     for i in range(lnc_length):
#         lnc_sim[i, i] = 1
#     np.savetxt('./data_create/lnc_sim.txt',lnc_sim)

# test
l = [[0,1,1,1],
     [0,0,0,0],
     [0,1,0,1],
     [0,0,0,0]]
d = [[1,0.1,0.2,0.3],
     [0.1,1,0.4,0.5],
     [0.2,0.4,1,0.6],
     [0.3,0.5,0.6,1]]

def get_all_sim1(lnc_dis,dis_sim):
    lnc_length = len(lnc_dis)
    lnc_sim = np.zeros((lnc_length,lnc_length))
    print(np.shape(lnc_sim))

    for i in range(lnc_length):
        for j in range(lnc_length):
            sim = get_sim_value(lnc_dis[i],lnc_dis[j],dis_sim)
            lnc_sim[i,j] = sim
    for i in range(lnc_length):
        lnc_sim[i, i] = 1
    print(lnc_sim)


l = np.array(l)
d = np.array(d)
# get_all_sim1(l,d)
l = [[0.1,1.1,1.2,1.2],
     [0,0,0,0],
     [0,1,0,1],
     [0,0,0,0]]


d = [[0,1,1,1],
     [0,0,0,0],
     [0,1,0,1],
     [0,0,0,0]]
l = np.array(l)
d = np.array(d)
c = np.where(l>2)
print(type(c))
print(c)
print(np.shape(c)[1])

if np.shape(c)[1]:
    print(1111)
