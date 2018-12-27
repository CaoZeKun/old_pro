import data_prepare as dp
import copy
import numpy as np


#2 class node
class node():
    def __init__(self, value, x, y):
        self.value = value  # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
        #self.miRNA = 0
        #self.disease = 0
        #self.predict_probability = 0.0                  #  row-mi  coloumn(mj+dj)  or row-di  coloumn(mj+dj)
        #self.predict_value = -1
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

#3 class matrix_A for its value and index
class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column


# the reason of using deep.copy is that
# in the last circulation temp_path_nodes will change
# so that in next circulation temp_path_nodes still restore the last result
def path_node2(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length

    miRNA_length = len(miRNA_miRNA_matrix)
    disease_length = len(disease_disease_matrix)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = []
    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    # path_edge_weights = []

    # for sample,from m[x] to d[y]
    # get index
    x = current_data.index_x  # miRNA
    y = current_data.index_y  # disease
    value = current_data.value
    # -----------for save in file-----------  #
    # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
    # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
    nodes_A = [x, y]
    # value_A = [value]
    path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
    # path_edge_weights.append(value_A)  # the first  value is A[x,y]

    # -----------first search sample i path length = 2-----------  #
    # print("---4.star search path---")
    # m[x] - m[j] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                temp_path_nodes.extend([600 + y])  # record point d[y]
                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                # temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                # permanent save
                path_nodes.append(temp_path_nodes)
                # path_edge_weights.append(temp_path_edge_weights)
    # print("---path m[x] - m[j] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                temp_path_nodes.extend([600 + y])  # record point d[y]
                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                # temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                # permanent save
                path_nodes.append(temp_path_nodes)
                # path_edge_weights.append(temp_path_edge_weights)
    # print("---path m[x] - d[j] - d[y],over---")
    # print(path_nodes)

    # -----------then search sample i path length = 3-----------  #
    # m[x] - m[j] - m[k] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []

        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(miRNA_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == x):
                        continue
                    elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                          current_associated_A[k][y] == 1):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([k])  # record point m[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([miRNA_miRNA_matrix[j][k]])  # record weight  m[j] - m[k]
                        # temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        path_nodes.append(temp_node)
                        # path_edge_weights.append(temp_weights)
    # print("---path m[x] - m[j] - m[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - m[j] - d[k] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []

        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == y):
                        continue
                    elif (current_associated_A[j][k] == 1 and
                          disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([current_associated_A[j][k]])  # record weight  m[j] - d[k]
                        # temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        path_nodes.append(temp_node)
                        # path_edge_weights.append(temp_weights)
    # print("---path m[x] - m[j] - d[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[k] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == y):
                        continue
                    if (disease_disease_matrix[j][k] > threshold_value and
                            disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([disease_disease_matrix[j][k]])  # record weight  d[j] - d[k]
                        # temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        path_nodes.append(temp_node)
                        # path_edge_weights.append(temp_weights)
    # m[x] - d[j] - m[k] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(miRNA_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == x):
                        continue
                    if (current_associated_A[k][j] == 1 and
                            current_associated_A[k][y] == 1):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([current_associated_A[k][j]])  # record weight  d[j] - m[k]
                        # temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        path_nodes.append(temp_node)
                        # path_edge_weights.append(temp_weights)
    # print("---looking path in current data, over")

    # print("---path m[x] - d[j] - d[k] - d[y],over---")    [374, 911, 758]

    return path_nodes


# search i - j - k  three nodes
def path_three_node(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length

    miRNA_length = len(miRNA_miRNA_matrix)
    disease_length = len(disease_disease_matrix)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = []
    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    path_edge_weights = []

    # for sample,from m[x] to d[y]
    # get index
    x = current_data.index_x  # miRNA
    y = current_data.index_y  # disease
    value = current_data.value
    # -----------for save in file-----------  #
    # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
    # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
    nodes_A = [x, y]
    value_A = [value]
    path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
    path_edge_weights.append(value_A)  # the first  value is A[x,y]

    # -----------first search sample i path length = 2-----------  #
    # print("---4.star search path---")
    # m[x] - m[j] - d[y]
    for j in range(miRNA_length):
        # temp_path_nodes = []
        #temp_path_edge_weights = []
        # record initial point
        # temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                # temporary save
                # temp_path_nodes.extend([j])  # record point m[j]
                # temp_path_nodes.extend([600 + y])  # record point d[y]
                #temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                #temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                # permanent save
                path_nodes.append([x,j,600+y])
                path_edge_weights.append([miRNA_miRNA_matrix[x][j],current_associated_A[j][y]])
    # print("---path m[x] - m[j] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[y]
    for j in range(disease_length):
        # temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point
        # temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                # temporary save
                # temp_path_nodes.extend([600 + j])  # record point d[j]
                # temp_path_nodes.extend([600 + y])  # record point d[y]
                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                # temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                # permanent save
                path_nodes.append([x, 600+j, 600 + y])
                path_edge_weights.append([current_associated_A[x][j],disease_disease_matrix[j][y]])
    # print("---path m[x] - d[j] - d[y],over---")
    # print(path_nodes)

    return path_nodes,path_edge_weights


# merge extend operator 寻找路径，利用路径信息过LSTM得到全局信息
def path_node2_new(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length

    miRNA_length = len(miRNA_miRNA_matrix)
    disease_length = len(disease_disease_matrix)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = []
    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    # path_edge_weights = []

    # for sample,from m[x] to d[y]
    # get index
    x = current_data.index_x  # miRNA
    y = current_data.index_y  # disease
    value = current_data.value
    # -----------for save in file-----------  #
    # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
    # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
    nodes_A = [x, y]
    # value_A = [value]
    path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
    # path_edge_weights.append(value_A)  # the first  value is A[x,y]

    # -----------first search sample i path length = 2-----------  #
    # print("---4.star search path---")
    # m[x] - m[j] - d[y]
    for j in range(miRNA_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point

        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                # temporary save
                #temp_path_nodes.extend([x])
                #temp_path_nodes.extend([j])  # record point m[j]
                #temp_path_nodes.extend([600 + y])  # record point d[y]
                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                # temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                # permanent save
                #path_nodes.append(temp_path_nodes)
                path_nodes.append([x,j,600+y])
                # path_edge_weights.append(temp_path_edge_weights)
    # print("---path m[x] - m[j] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[y]
    for j in range(disease_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point

        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                #temp_path_nodes.extend([x])
                # temporary save
                #temp_path_nodes.extend([600 + j])  # record point d[j]
                #temp_path_nodes.extend([600 + y])  # record point d[y]
                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                # temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                # permanent save
                #path_nodes.append(temp_path_nodes)
                path_nodes.append([x, 600+j, 600 + y])
                # path_edge_weights.append(temp_path_edge_weights)
    # print("---path m[x] - d[j] - d[y],over---")
    # print(path_nodes)

    # -----------then search sample i path length = 3-----------  #
    # m[x] - m[j] - m[k] - d[y]
    for j in range(miRNA_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []


        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save

                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(miRNA_length):
                    #temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    #point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == x):
                        continue
                    elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                          current_associated_A[k][y] == 1):
                        #point_added_flag = 1
                        # temporary save
                        # record initial point
                        #temp_path_nodes.extend([x])
                        #temp_path_nodes.extend([j])  # record point m[j]
                        #temp_path_nodes.extend([k])  # record point m[k]
                        #temp_path_nodes.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([miRNA_miRNA_matrix[j][k]])  # record weight  m[j] - m[k]
                        # temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        #path_nodes.append(temp_path_nodes)
                        path_nodes.append([x, j, k,600 + y])
                        # path_edge_weights.append(temp_weights)
    # print("---path m[x] - m[j] - m[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - m[j] - d[k] - d[y]
    for j in range(miRNA_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []

        # record initial point

        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save

                # temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(disease_length):
                    #temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    #point_added_flag = 0  # judging the point added in paths or not
                    if (k == y):
                        continue
                    elif (current_associated_A[j][k] == 1 and
                          disease_disease_matrix[k][y] > threshold_value):
                        #point_added_flag = 1
                        # temporary save
                        #temp_path_nodes.extend([x])
                        #temp_path_nodes.extend([j])  # record point m[j]

                        #temp_path_nodes.extend([600 + k])  # record point d[k]
                        #temp_path_nodes.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([current_associated_A[j][k]])  # record weight  m[j] - d[k]
                        # temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        #path_nodes.append(temp_path_nodes)
                        path_nodes.append([x, j, 600+k, 600 + y])

                        # path_edge_weights.append(temp_weights)
    # print("---path m[x] - m[j] - d[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[k] - d[y]
    for j in range(disease_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point

        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                # temporary save

                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(disease_length):
                    #temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    #point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == y):
                        continue
                    if (disease_disease_matrix[j][k] > threshold_value and
                            disease_disease_matrix[k][y] > threshold_value):
                        #point_added_flag = 1
                        # temporary save
                        #temp_path_nodes.extend([x])
                        #temp_path_nodes.extend([600 + j])  # record point d[j]
                        #temp_path_nodes.extend([600 + k])  # record point d[k]
                        #temp_path_nodes.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([disease_disease_matrix[j][k]])  # record weight  d[j] - d[k]
                        # temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        #path_nodes.append(temp_path_nodes)
                        path_nodes.append([x, 600+j, 600+k, 600 + y])
                        # path_edge_weights.append(temp_weights)
    # m[x] - d[j] - m[k] - d[y]
    for j in range(disease_length):
        #temp_path_nodes = []
        # temp_path_edge_weights = []
        # record initial point

        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                # temporary save

                # temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(miRNA_length):
                    #temp_node = copy.deepcopy(temp_path_nodes)
                    # temp_weights = copy.deepcopy(temp_path_edge_weights)
                    #point_added_flag = 0  # judging the point added in paths or not
                    if (k == x):
                        continue
                    if (current_associated_A[k][j] == 1 and
                            current_associated_A[k][y] == 1):
                        #point_added_flag = 1
                        # temporary save
                        #temp_path_nodes.extend([x])
                        #temp_path_nodes.extend([600 + j])  # record point d[j]
                        #temp_path_nodes.extend([k])  # record point d[k]
                        #temp_path_nodes.extend([600 + y])  # record point d[y]
                        # temp_weights.extend([current_associated_A[k][j]])  # record weight  d[j] - m[k]
                        # temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        #path_nodes.append(temp_path_nodes)
                        path_nodes.append([x, 600+j, k, 600 + y])
                        # path_edge_weights.append(temp_weights)
    # print("---looking path in current data, over")

    # print("---path m[x] - d[j] - d[k] - d[y],over---")    [374, 911, 758]

    return path_nodes

# merge extend operator but no x,y(goal) in path except the list where save gol([miRna, disease])
# 1.只保留路径节点信息即保留共性，有共同路径节点得可能性更大，CNN扫描？/LSTM
# 2.用goal[x,y]保留目标信息即保留局部特性
def path_node2_new2(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length

    miRNA_length = len(miRNA_miRNA_matrix)
    disease_length = len(disease_disease_matrix)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = []
    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    # path_edge_weights = []

    # for sample,from m[x] to d[y]
    # get index
    x = current_data.index_x  # miRNA
    y = current_data.index_y  # disease
    value = current_data.value
    # -----------for save in file-----------  #
    # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
    # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
    nodes_A = [x, y]
    # value_A = [value]
    path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
    # path_edge_weights.append(value_A)  # the first  value is A[x,y]

    # -----------first search sample i path length = 2-----------  #
    # print("---4.star search path---")
    # m[x] - m[j] - d[y]
    for j in range(miRNA_length):
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                path_nodes.append([j])
                # path_edge_weights.append(temp_path_edge_weights)
    # print("---path m[x] - m[j] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[y]
    for j in range(disease_length):
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                path_nodes.append([600 + j])
    # print("---path m[x] - d[j] - d[y],over---")
    # print(path_nodes)

    # -----------then search sample i path length = 3-----------  #
    # m[x] - m[j] - m[k] - d[y]
    for j in range(miRNA_length):
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                for k in range(miRNA_length):
                    if (k == j or k == x):
                        continue
                    elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                          current_associated_A[k][y] == 1):
                        path_nodes.append([ j, k])
    # print("---path m[x] - m[j] - m[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - m[j] - d[k] - d[y]
    for j in range(miRNA_length):
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):

                for k in range(disease_length):

                    if (k == y):
                        continue
                    elif (current_associated_A[j][k] == 1 and
                          disease_disease_matrix[k][y] > threshold_value):
                        path_nodes.append([j, 600 + k])
    # print("---path m[x] - m[j] - d[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[k] - d[y]
    for j in range(disease_length):
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                for k in range(disease_length):
                    if (k == j or k == y):
                        continue
                    if (disease_disease_matrix[j][k] > threshold_value and
                            disease_disease_matrix[k][y] > threshold_value):

                        path_nodes.append([600 + j, 600 + k])
    # m[x] - d[j] - m[k] - d[y]
    for j in range(disease_length):
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                for k in range(miRNA_length):
                    if (k == x):
                        continue
                    if (current_associated_A[k][j] == 1 and
                            current_associated_A[k][y] == 1):
                        path_nodes.append([600 + j, k])
    # print("---looking path in current data, over")

    # print("---path m[x] - d[j] - d[k] - d[y],over---")    [374, 911, 758]

    return path_nodes

def save_paths_nodes3(train_set,dev_set,test_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value=0):  # just for one validation
    print("---4.star search path---")
    # save train set
    length_train_set = len(train_set)
    path_node_train_set = []
    #path_edge_weights_train_set = []
    for i in range(length_train_set):#7124
        # temp_path_nodes = path_node2(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
        temp_path_nodes = path_three_node(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix,
                                     threshold_value)
        path_node_train_set.append(temp_path_nodes)
        #path_edge_weights_train_set.append(temp_path_edge_weights)
        if(i == 7123):
            np.savez('./get_data/path_search/train/train_nodes' + str(i + 1), train_set=path_node_train_set)
            #np.savez('./get_data/path_search/train/train_weights' + str(i + 1), train_set=path_edge_weights_train_set)
            path_node_train_set = []
            #path_edge_weights_train_set = []
            print("---Train data path search over")
        if((i+1)%500==0):
            print("---This is %d in train set"%(i))
            np.savez('./get_data/path_search/train/train_nodes'+str(i+1), train_set=path_node_train_set)
            #np.savez('./get_data/path_search/train/train_weights'+str(i+1), train_set=path_edge_weights_train_set)
            path_node_train_set = []
            #path_edge_weights_train_set = []

    # save dev set
    length_dev_set = len(dev_set)
    path_node_dev_set = []
    #path_edge_weights_dev_set = []
    for i in range(length_dev_set):#2036
        # temp_path_nodes = path_node2(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
        #                                                     disease_disease_matrix, threshold_value)
        temp_path_nodes = path_three_node(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
                                     disease_disease_matrix, threshold_value)
        path_node_dev_set.append(temp_path_nodes)
        #path_edge_weights_dev_set.append(temp_path_edge_weights)
        if (i == 2035):
            np.savez('./get_data/path_search/dev/dev_nodes' + str(i + 1), dev_set=path_node_dev_set)
            #np.savez('./get_data/path_search/dev/dev_weights' + str(i + 1), dev_set=path_edge_weights_dev_set)
            path_node_dev_set = []
            #path_edge_weights_dev_set = []
            print("---Train data path search over")
        if ((i + 1) % 500 == 0):
            print("---This is %d in dev set" % (i))
            np.savez('./get_data/path_search/dev/dev_nodes' + str(i+1), dev_set=path_node_dev_set)
            #np.savez('./get_data/path_search/dev/dev_weights' + str(i+1), dev_set=path_edge_weights_dev_set)
            path_node_dev_set = []
            #path_edge_weights_dev_set = []

    # save test set
    length_test_set = len(test_set)
    path_node_test_set = []
    path_edge_weights_test_set = []
    for i in range(length_test_set):#155160
        # temp_path_nodes = path_node2(test_set[i], current_associated_A, miRNA_miRNA_matrix,
        #                                                     disease_disease_matrix, threshold_value)
        temp_path_nodes = path_three_node(test_set[i], current_associated_A, miRNA_miRNA_matrix,
                                     disease_disease_matrix, threshold_value)
        path_node_test_set.append(temp_path_nodes)
        #path_edge_weights_test_set.append(temp_path_edge_weights)
        if (i == 155159):
            np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
            #np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
            path_node_test_set = []
            #path_edge_weights_test_set = []
            print("---Train data path search over")
        if ((i + 1) % 500 == 0):
            print("---This is %d in test set" % (i))
            np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
            #np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
            path_node_test_set = []
            #path_edge_weights_test_set = []

# use one list save all sets
def save_paths_nodes4(train_set,dev_set,test_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, k_count,threshold_value=0):  # just for one validation
    print("---4.star search path---")
    # save train set
    length_train_set = len(train_set)
    path_node_train_set = []
    path_edge_weights_train_set = []
    for i in range(length_train_set):#7124
        # temp_path_nodes = path_node2(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
        temp_path_nodes,temp_path_edge_weights = path_three_node(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix,
                                     threshold_value)
        path_node_train_set.append(temp_path_nodes)
        path_edge_weights_train_set.append(temp_path_edge_weights)
        # path_edge_weights_train_set.append(temp_path_edge_weights)
        # if(i == 7123):
        #     np.savez('./get_data/path_search/train/train_nodes' + str(i + 1), train_set=path_node_train_set)
    np.savez('./get_data/path_search/new_all/train_nodes' + str(k_count), path_node_train_set=path_node_train_set)
        #     #np.savez('./get_data/path_search/train/train_weights' + str(i + 1), train_set=path_edge_weights_train_set)
        #     path_node_train_set = []
        #     #path_edge_weights_train_set = []
        #     print("---Train data path search over")
        # if((i+1)%500==0):
        #     print("---This is %d in train set"%(i))
        #     np.savez('./get_data/path_search/train/train_nodes'+str(i+1), train_set=path_node_train_set)
        #     #np.savez('./get_data/path_search/train/train_weights'+str(i+1), train_set=path_edge_weights_train_set)
        #     path_node_train_set = []
        #     #path_edge_weights_train_set = []

    # save dev set
    length_dev_set = len(dev_set)
    path_node_dev_set = []
    path_edge_weights_dev_set = []
    for i in range(length_dev_set):#2036
        # temp_path_nodes = path_node2(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
        #                                                     disease_disease_matrix, threshold_value)
        temp_path_nodes,temp_path_edge_weights = path_three_node(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
                                     disease_disease_matrix, threshold_value)
        path_node_dev_set.append(temp_path_nodes)
        path_edge_weights_dev_set.append(temp_path_edge_weights)
        # path_edge_weights_dev_set.append(temp_path_edge_weights)
        # if (i == 2035):
        #     np.savez('./get_data/path_search/dev/dev_nodes' + str(i + 1), dev_set=path_node_dev_set)
    np.savez('./get_data/path_search/new_all/dev_nodes' + str(k_count), path_node_dev_set=path_node_dev_set)
        #     #np.savez('./get_data/path_search/dev/dev_weights' + str(i + 1), dev_set=path_edge_weights_dev_set)
        #     path_node_dev_set = []
        #     #path_edge_weights_dev_set = []
        #     print("---Train data path search over")
        # if ((i + 1) % 500 == 0):
        #     print("---This is %d in dev set" % (i))
        #     np.savez('./get_data/path_search/dev/dev_nodes' + str(i+1), dev_set=path_node_dev_set)
        #     #np.savez('./get_data/path_search/dev/dev_weights' + str(i+1), dev_set=path_edge_weights_dev_set)
        #     path_node_dev_set = []
        #     #path_edge_weights_dev_set = []

    # save test set
    length_test_set = len(test_set)
    path_node_test_set = []
    path_edge_weights_test_set = []
    for i in range(length_test_set):#155160
        # temp_path_nodes = path_node2(test_set[i], current_associated_A, miRNA_miRNA_matrix,
        #                                                     disease_disease_matrix, threshold_value)
        temp_path_nodes,temp_path_edge_weights = path_three_node(test_set[i], current_associated_A, miRNA_miRNA_matrix,
                                     disease_disease_matrix, threshold_value)
        path_node_test_set.append(temp_path_nodes)
        path_edge_weights_test_set.append(temp_path_edge_weights)
        # path_edge_weights_test_set.append(temp_path_edge_weights)
        # if (i == 155159):
        #     np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
    np.savez('./get_data/path_search/new_all/test_nodes' + str(k_count), path_node_test_set=path_node_test_set)
        #     #np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
        #     path_node_test_set = []
        #     #path_edge_weights_test_set = []
        #     print("---Train data path search over")
        # if ((i + 1) % 500 == 0):
        #     print("---This is %d in test set" % (i))
        #     np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
        #     #np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
        #     path_node_test_set = []
        #     #path_edge_weights_test_set = []
    return path_node_train_set,path_node_dev_set,path_node_test_set,path_edge_weights_train_set,path_edge_weights_dev_set,path_edge_weights_test_set
if __name__ == '__main__':
    A, Sm, Sd = dp.read_data_flies()
    k = 10
    D = np.load('./get_data/data_prepare/devided_data.npz')
    all_k_train_set = D['all_k_train_set']
    all_k_development_set = D['all_k_development_set']
    all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']


    D = np.load('./get_data/data_prepare/devided_data.npz')
    all_k_train_set = D['all_k_train_set']

    all_k_development_set = D['all_k_development_set']
    all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']

    # save_paths_nodes3(all_k_train_set[0],all_k_development_set[0],all_k_test_set[0],save_all_count_A[0],Sm,Sd, threshold_value=0.2)
    path_node_train_set, path_node_dev_set, path_node_test_set, path_edge_weights_train_set, path_edge_weights_dev_set, path_edge_weights_test_set = save_paths_nodes4(all_k_train_set[0], all_k_development_set[0], all_k_test_set[0], save_all_count_A[0], Sm, Sd,
                      0,threshold_value=0.2)
    print(path_edge_weights_dev_set)
    # for i in range(k):
    #     save_paths_nodes4(all_k_train_set[i], all_k_development_set[i], all_k_test_set[i], save_all_count_A[i], Sm, Sd,
    #                   i,threshold_value=0.2)
    # for i in range(1000):
    # path_nodes, path_edge_weights = path_node(all_k_train_set[0][i], save_all_count_A[0], Sm, Sd, 0)
    # print(path_nodes[-1])