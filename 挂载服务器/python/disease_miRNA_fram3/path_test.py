import data_prepare as dp
import copy
import numpy as np




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




SM = [[1,0.2,0.6],
      [0.2,1,0.3],
      [0.6,0.3,1]]

SD = [[1,0.3,0.2],
      [0.3,1,0.5],
      [0.2,0.5,1]]

A = [[1,0,1],
     [0,0,1],
     [1,1,0]]

class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.index_x = i  # save index of row
        self.index_y = j  # save index of column


p = []

for i in range(len(A)):
    for j in range(len(A[0])):
        node = value_index(A[i][j],i,j)
        p.append(node)

path_node = path_node2(p[1],A,SM,SD,0.2)
print(path_node)
