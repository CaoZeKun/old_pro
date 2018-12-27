import data_prepare as dp
import path_search as ps
import numpy as np


#2 class node
class node():
    def __init__(self, value, x, y):
        self.value = value  # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)  miRNA
        self.index_y = y  # the column in A_association A(x,y)  disease
        self.miRNA = 0
        self.disease = 0
        self.predict_probability = 0.0                  #  row-mi  coloumn(mj+dj)  or row-di  coloumn(mj+dj)
        self.predict_value = -1
    def judge_RNA_disease(self,flag):
        if(flag == 1):
            self.miRNA = 1
        else:
            self.disease =1
    def add_association(self,m_d_association):
        self.m_d_association = m_d_association  # 1 * 490 + 1 * 326 = 1 * 816     1 * (miRNA + disease)
                                                #  row-mi  coloumn(mj+dj)  or row-di  coloumn(mj+dj)
    def add_predict_probability_value(self,predict_probability,predict_value):
        self.predict_probability = predict_probability
        self.predict_value = predict_value


#8 old read npz data
def read_npz_data():
    # read data
    current_A = np.load('current_A.npz')
    nodes = np.load('nodes.npz')
    weights = np.load('weights.npz')

    # train set-----7124
    nodes_train_set = nodes['train_set']
    weights_train_set = weights['train_set']

    # dev set------2036
    nodes_dev_set = nodes['dev_set']
    weights_dev_set = weights['dev_set']

    #test set-----155160 contain all 0
    nodes_test_set = nodes['test_set']
    weights_test_set = weights['test_set']
    print("---read data npz")

    return nodes_train_set,weights_train_set,nodes_dev_set,weights_dev_set,nodes_test_set,weights_test_set,current_A
#----------------------------------------------------------------

#8 read npz data
def read_npz_data1(flag,i):
    # read data
    if(flag == 1):  # train set
        nodes = np.load('./get_data/path_search/train/train_nodes'+str(i)+('.npz'))['train_set']
    if(flag == 2):  #dev set
        nodes = np.load('./get_data/path_search/dev/dev_nodes' + str(i) + ('.npz'))['dev_set']
    if(flag == 3):  #test data
        nodes = np.load('./get_data/path_search/test/test_nodes' + str(i) + ('.npz'))['test_set']
    return nodes

#  old add feature to node in path
def add_feature_node1(data,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    data_length = len(data)
    data_feature = []
    for i in range(data_length):
        if ( data[i] >= 600): # di [mj dj]
            temp_d_m = current_associated_A[:,(data[i]-600)]
            temp_d_d = disease_disease_matrix[(data[i]-600)]
            temp_feature = np.concatenate((temp_d_m,temp_d_d))
            data_feature.append(temp_feature)
        else:
            temp_m_m = miRNA_miRNA_matrix[i]
            temp_m_d = current_associated_A[i]
            temp_feature = np.concatenate((temp_m_m,temp_m_d))
            data_feature.append(temp_feature)

    #print(data_feature) [[] [] []]
    return data_feature


# old add feature to node in path
def add_feature_node3(data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    data_length = len(data)
    data_feature = []
    data_feature_length = [data_length]
    for i in range(data_length):
        if (data[i] >= 600):  # di [mj dj]
            temp_d_m = current_associated_A[:, (data[i] - 600)]
            temp_d_d = disease_disease_matrix[(data[i] - 600)]
            temp_feature = np.concatenate((temp_d_m, temp_d_d))
            data_feature.append(temp_feature)
        else:
            temp_m_m = miRNA_miRNA_matrix[i]
            temp_m_d = current_associated_A[i]
            temp_feature = np.concatenate((temp_m_m, temp_m_d))
            data_feature.append(temp_feature)
    #padding for not enough 4 nodes in the path
    #a = np.zeros((816))
    if(data_length < 4):
        lack_node_length = 4 - data_length
        a = np.zeros((lack_node_length,816))
        data_feature = np.concatenate((data_feature, a))
    # print(data_feature) [[] [] [] []]
    return data_feature,data_feature_length
#  add feature to node in path ,not padding in path
def add_feature_node(data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    data_length = len(data)
    data_feature = []
    data_feature_length = [data_length]
    for i in range(data_length):
        if (data[i] >= 600):  # di [mj dj]
            temp_d_m = current_associated_A[:, (data[i] - 600)]
            temp_d_d = disease_disease_matrix[(data[i] - 600)]
            temp_feature = np.concatenate((temp_d_m, temp_d_d))
            data_feature.append(temp_feature)
        else:
            temp_m_m = miRNA_miRNA_matrix[i]
            temp_m_d = current_associated_A[i]
            temp_feature = np.concatenate((temp_m_m, temp_m_d))
            data_feature.append(temp_feature)
    #padding for not enough 4 nodes in the path
    #a = np.zeros((816))
    #if(data_length < 4):
        #lack_node_length = 4 - data_length
        #a = np.zeros((lack_node_length,816))
        #data_feature = np.concatenate((data_feature, a))
    # print(data_feature) [[] [] [] []]
    return data_feature,data_feature_length


#  add feature to node in path ,not padding in path
def add_feature_one_node(data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    data_feature = []
    if (data >= 600):  # di [mj dj]
        temp_d_m = current_associated_A[:, (data - 600)]
        temp_d_d = disease_disease_matrix[(data - 600)]
        temp_feature = np.concatenate((temp_d_m, temp_d_d))
        data_feature.extend(temp_feature)
    else:
        temp_m_m = miRNA_miRNA_matrix[data]
        temp_m_d = current_associated_A[data]
        temp_feature = np.concatenate((temp_m_m, temp_m_d))
        data_feature.extend(temp_feature)

    #padding for not enough 4 nodes in the path
    #a = np.zeros((816))
    #if(data_length < 4):
        #lack_node_length = 4 - data_length
        #a = np.zeros((lack_node_length,816))
        #data_feature = np.concatenate((data_feature, a))
    # print(data_feature) [[] [] [] []]
    return data_feature

# add feature for start noed and end node
def add_feature_goal_node(data_x,data_y,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    #data_feature = []
    temp_d_m = current_associated_A[:,data_y]
    temp_d_d = disease_disease_matrix[data_y]
    temp_feature1 = np.concatenate((temp_d_m,temp_d_d))
    #data_feature.append(temp_feature)
    temp_m_m = miRNA_miRNA_matrix[data_x]
    temp_m_d = current_associated_A[data_x]
    temp_feature2 = np.concatenate((temp_m_m,temp_m_d))
    #data_feature.append(temp_feature)
    data_feature = np.concatenate((temp_feature2,temp_feature1))

    #print(data_feature) [[]
                        # []]
    return data_feature,temp_feature1,temp_feature2

#9  add feature to every single sample (nodes,path)---padding 0, if length is not enough in all path length or in the path which nodes not enough 4
def add_feature(nodes_set,matrix_A,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    data_length = len(nodes_set)
    pre_padding_length  = []
    for i in range(data_length):
        temp_path_feature = []
        for j in range(len(nodes_set[i])):
            temp = []
            if(j==0):
                temp.extend(nodes_set[i][0])
                temp.extend([matrix_A[nodes_set[i][0][0]][nodes_set[i][0][1]]])
                save_for_goals.append(temp)
            else:
                temp_feature = add_feature_node(nodes_set[i][j],current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix)
                temp_path_feature.append(temp_feature)

        #if current nodes_set[i] is not reach longest length in all nodes_set,padding
        if(len(nodes_set[i]) <= 103520):
            tem_length = len(nodes_set[i])
            lack_path_length = 103520 - tem_length
            list_2d = [[[0 for row in range(816)] for row in range(4)] for row in range(lack_path_length)]
            temp_path_feature.extend(list_2d)
            #print(temp_path_feature)

            pre_padding_length.extend([tem_length])
            #temp_path_feature = np.concatenate((temp_path_feature,a))
        print(np.shape(temp_path_feature))
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
    save_for_goals = np.array(save_for_goals)
    save_for_paths_feature = np.array(save_for_paths_feature)
    pre_padding_length = np.array(pre_padding_length)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature,pre_padding_length

#9  add feature to every single sample (nodes,path)---padding 0, if length is not enough in all path length or in the path which nodes not enough 4
def add_feature3(nodes_set,matrix_A,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    data_length = len(nodes_set)
    pre_padding_length  = []
    for i in range(data_length):
        temp_path_feature = []
        for j in range(len(nodes_set[i])):
            temp = []
            if(j==0):
                temp.extend(nodes_set[i][0])
                temp.extend([matrix_A[nodes_set[i][0][0]][nodes_set[i][0][1]]])
                save_for_goals.append(temp)
            else:
                temp_feature = add_feature_node3(nodes_set[i][j],current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix)
                temp_path_feature.append(temp_feature)

        #if current nodes_set[i] is not reach longest length in all nodes_set,padding
        if(len(nodes_set[i]) <= 103520):
            tem_length = len(nodes_set[i])
            lack_path_length = 103520 - tem_length
            a = np.zeros((lack_path_length,4,816))
            #list_2d = [[[0 for row in range(816)] for row in range(4)] for row in range(lack_path_length)]
            #temp_path_feature.extend(list_2d)
            #print(temp_path_feature)

            pre_padding_length.extend([tem_length])
            temp_path_feature = np.concatenate((temp_path_feature,a))
        print(np.shape(temp_path_feature))
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
    save_for_goals = np.array(save_for_goals)
    save_for_paths_feature = np.array(save_for_paths_feature)
    pre_padding_length = np.array(pre_padding_length)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature,pre_padding_length


def add_feature4(nodes_set,matrix_A,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    save_all_paths_length = []
    data_length = len(nodes_set)
    #pre_padding_length  = []
    for i in range(data_length):
        temp_path_feature = []
        temp_path_feature_length = []
        for j in range(len(nodes_set[i])):
            temp = []
            if(j==0):
                temp.extend(nodes_set[i][0])
                temp.extend([matrix_A[nodes_set[i][0][0]][nodes_set[i][0][1]]])
                save_for_goals.append(temp)
            else:
                temp_feature,data_feature_length = add_feature_node(nodes_set[i][j],current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix)
                temp_path_feature.append(temp_feature)
                temp_path_feature_length.extend(data_feature_length)
        save_all_paths_length.append(temp_path_feature_length)
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
    save_for_goals = np.array(save_for_goals)
    save_for_paths_feature = np.array(save_for_paths_feature)
    save_all_paths_length = np.array(save_all_paths_length)
    #pre_padding_length = np.array(pre_padding_length)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature,save_all_paths_length


def add_padding():
    a = np.zeros((1, 816))
    return a
# nodes_set only one set
def add_feature5(nodes_set,matrix_A,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_goal_nodes = []
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    save_all_paths_length = []
    data_length = len(nodes_set)
    #pre_padding_length  = []


    #save_for_goals.append(temp)
    for path in range(data_length-1,-1,-1):
        #print(path)
        temp_path_feature = []
        temp = []
        if (path == 0):
            temp.extend(nodes_set[path])
            # print(temp)
            # print([matrix_A[nodes_set[i][0]][nodes_set[i][1]]])
            temp.extend([matrix_A[nodes_set[path][0]][nodes_set[path][1]]])
            # print(temp)
            save_for_goals.extend(temp)
        #temp_path_feature_length = []
        for node in range(len(nodes_set[path])):
                # print(nodes_set[i][j])

            temp_feature = add_feature_one_node(nodes_set[path][node], current_associated_A, miRNA_miRNA_matrix,
                                                    disease_disease_matrix)
                # print(np.shape(temp_feature))
            temp_path_feature.append(temp_feature)
            #print(np.shape(temp_path_feature))
        if (len(nodes_set[path]) < 3):
            padding_0 = add_padding()
            temp_path_feature.extend(padding_0)
            #print(np.shape(temp_path_feature))

                # temp_path_feature_length.extend(data_feature_length)
        #print(len(temp_path_feature))
        save_all_paths_length.extend([len(nodes_set[path])])
        #print(save_all_paths_length)
        #save_all_paths_length.append(temp_path_feature_length)
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
        #print("1111")
        #print(np.shape(save_for_paths_feature))
    save_for_goals = np.array(save_for_goals)
    #save_for_goal_nodes.append(nodes_set[0])
    #print(save_for_goal_nodes)
    save_for_paths_feature = np.array(save_for_paths_feature)
    #save_all_paths_length = np.array(save_all_paths_length)
    #pre_padding_length = np.array(pre_padding_length)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature,save_all_paths_length  # paths * nodes * feature

#9old  add feature to every single sample (nodes,path)
def add_feature2(nodes_set,matrix_A,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    data_length = len(nodes_set)
    for i in range(data_length):
        temp_path_feature = []
        for j in range(len(nodes_set[i])):
            temp = []
            if(j==0):
                temp.extend(nodes_set[i][0])
                temp.extend([matrix_A[nodes_set[i][0][0]][nodes_set[i][0][1]]])
                save_for_goals.append(temp)
            else:
                temp_feature = add_feature_node(nodes_set[i][j],current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix)
                temp_path_feature.append(temp_feature)
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
    save_for_goals = np.array(save_for_goals)
    save_for_paths_feature = np.array(save_for_paths_feature)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature

#9old old add feature to every single sample (nodes,path)
def add_feature1(nodes_set,weights_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix):
    save_for_goals = []  #[[mi,dj,value]  ]
    save_for_paths_feature = []  # [  [sample[path_i[][][]]            ]
    data_length = len(nodes_set)
    for i in range(data_length):
        temp_path_feature = []
        for j in range(len(nodes_set[i])):
            temp = []
            if(j==0):
                temp.extend(nodes_set[i][0])
                temp.extend(weights_set[i][0])
                save_for_goals.append(temp)
            else:
                temp_feature = add_feature_node(nodes_set[i][j],current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix)
                temp_path_feature.append(temp_feature)
        save_for_paths_feature.append(temp_path_feature)  # [ sample [path[node[] [] []]]]
    save_for_goals = np.array(save_for_goals)
    save_for_paths_feature = np.array(save_for_paths_feature)
    #print(np.shape(save_for_goals)) #(7124, 3)
    #print(np.shape(save_for_paths_feature)) #(7124,)
    return save_for_goals,save_for_paths_feature


def get_certain_data(nodes_set, weights_set, curren_A, Sm, Sd,flag=0):
    if(flag==1):
        #train data
        train_goals, train_paths_feature = add_feature(nodes_set, weights_set, curren_A, Sm, Sd)
        np.savez('./get_data/train', train_goals=train_goals, train_paths_feature=train_paths_feature)
        print("---train data over")
    if(flag==2):
        # dev data
        dev_goals, dev_paths_feature = add_feature(nodes_set, weights_set, curren_A, Sm, Sd)
        np.savez('./get_data/dev', dev_goals=dev_goals, dev_paths_feature=dev_paths_feature)
        print("---dev data over")
    if(flag==3):
        # test data
        test_goals, test_paths_feature = add_feature(nodes_set, weights_set, curren_A, Sm, Sd)
        np.savez('./get_data/test', test_goals=train_goals, test_paths_feature=train_paths_feature)
        print("---test data over")

#10 get goal and feature for train dev test set
def get_goal_feature_data():
    print("---5.get goal and feature to model---")
    nodes_train_set, weights_train_set, nodes_dev_set, weights_dev_set, nodes_test_set, weights_test_set, curren_A = read_npz_data()
    A, Sm, Sd = dp.read_data_flies()
    # train data
    get_certain_data(nodes_train_set, weights_train_set, curren_A, Sm, Sd,1)
    # dev data
    get_certain_data(nodes_dev_set, weights_dev_set, curren_A, Sm, Sd, 2)
    # test data
    get_certain_data(nodes_test_set, weights_test_set, curren_A, Sm, Sd, 3)







if __name__ == '__main__':
    # define parameters
    k = 10
    #all_k_train_set, all_k_development_set, all_k_test_set, all_k_count_positive, A, num, A_length, sava_association_A, Sm, Sd,save_all_count_A = dp.get_pre_train_dev_test_set(k)
    #ps.save_paths_nodes(all_k_train_set[0],all_k_development_set[0],all_k_test_set[0],save_all_count_A[0],Sm,Sd, threshold_value=0)
    #nodes_train_set, weights_train_set, nodes_dev_set, weights_dev_set, nodes_test_set, weights_test_set = read_npz_data()
    #print(weights_train_set[0])
    #print(nodes_train_set[0])  # format like[[133, 259], [133, 90, 859], [133, 105, 859], [133, 111, 859], [133, 172, 859], [133, 183, 859], [133, 457, 859]]
    nodes_set = read_npz_data1(1,500)
    #print(np.shape(nodes_set))   (500,)
    #print(len(nodes_set))  500
    #print(len(nodes_set[0]))  142
    #print(nodes_set[0])  [[254, 260], [254, 2, 826, 860], [ ]     ]

    A, Sm, Sd = dp.read_data_flies()
    D = np.load('./get_data/data_prepare/devided_data.npz')
    #all_k_train_set = D['all_k_train_set']
    #all_k_development_set = D['all_k_development_set']
    #all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']

    #data_feature = add_feature_node(nodes_set[1][1],save_all_count_A[0],Sm,Sd)
    #print(np.shape(data_feature))  # (4, 816)
    #print(data_feature[0])
    #print(data_feature[-1])
    #print(data_feature)

    goals, paths_feature = add_feature(nodes_set[:2],A,save_all_count_A[0],Sm,Sd)
    print(goals)
    print(np.shape(goals))#(10, 3)
    print(goals[0])#[ 254.  260.    0.]
    print(np.shape(paths_feature))
    #print(paths_feature[1][1])
    print(np.shape((paths_feature[1][1])))#(3, 816) #(4,816)
    print(len(paths_feature[0])) #103519
    print(np.shape(paths_feature[0]))#(141, 4, 816) #(103519,4,816)
    #train_goals, train_paths_feature, dev_goals, dev_paths_feature, test_goals, test_paths_feature = get_goal_feature_data()
