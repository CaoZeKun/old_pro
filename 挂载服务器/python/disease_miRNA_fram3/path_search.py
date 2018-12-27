import data_prepare as dp
import copy
import numpy as np

#2 class node
class node():
    def __init__(self, value, x, y):
        self.value = value  # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
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


#6 according to current Association Matrix, search points path  for all sample
def paths_nodes(current_data_set,current_associated_A,miRNA_miRNA_matrix,disease_disease_matrix,threshold_value):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length
    data_length = len(current_data_set)
    miRNA_length = len(miRNA_miRNA_matrix)
    disease_length = len(disease_disease_matrix)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]  
                               [sample2[path1 nodes]  [path2 nodes] ... [pathM nodes]]
                                ...
                               [sampleN[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = [[] for row in range(data_length)]
    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    path_edge_weights = [[] for row in range(data_length)]
    print("---4.star search path---")
    # for every sample,from m[x] to d[y]
    for i in range(data_length):
        # get index
        x = current_data_set[i].index_x  #miRNA
        y = current_data_set[i].index_y  #disease
        value = current_data_set[i].value
        # -----------for save in file-----------  #
        # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
        # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
        nodes_A = [x, y]
        value_A = [value]
        path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
        path_edge_weights.append(value_A)  # the first  value is A[x,y]

        # -----------first search sample i path length = 2-----------  #
        # m[x] - m[j] - d[y]
        for j in range(miRNA_length):
            temp_path_nodes = []
            temp_path_edge_weights = []
            # record initial point
            temp_path_nodes.extend([x])
            if(j == x):
                continue
            else:
                if(miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                    # temporary save
                    temp_path_nodes.extend([j])  # record point m[j]
                    temp_path_nodes.extend([600+y])  # record point d[y]
                    temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                    temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                    # permanent save
                    path_nodes[i].append(temp_path_nodes)
                    path_edge_weights[i].append(temp_path_edge_weights)
        #print("---path m[x] - m[j] - d[y],over---")
        #print(path_nodes)
        # m[x] - d[j] - d[y]
        for j in range(disease_length):
            temp_path_nodes = []
            temp_path_edge_weights = []
            # record initial point
            temp_path_nodes.extend([x])
            if (j == y):
                continue
            else:
                if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                    # temporary save
                    temp_path_nodes.extend([600 + j])  # record point d[j]
                    temp_path_nodes.extend([600 + y])  # record point d[y]
                    temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                    temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                    # permanent save
                    path_nodes[i].append(temp_path_nodes)
                    path_edge_weights[i].append(temp_path_edge_weights)
        #print("---path m[x] - d[j] - d[y],over---")
        # print(path_nodes)

        # -----------then search sample i path length = 3-----------  #
        # m[x] - m[j] - m[k] - d[y]
        for j in range(miRNA_length):
            temp_path_nodes = []
            temp_path_edge_weights = []
            point_added_flag = 0  # judging the point added in paths or not
            # record initial point
            temp_path_nodes.extend([x])
            if (j == x):
                continue
            else:
                if (miRNA_miRNA_matrix[x][j] > threshold_value ):
                    # temporary save
                    temp_path_nodes.extend([j])  # record point m[j]
                    temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(miRNA_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    if(k == j or k == x):
                        continue
                    elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                                current_associated_A[k][y] == 1):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([k])  # record point m[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([miRNA_miRNA_matrix[j][k]])  # record weight  m[j] - m[k]
                        temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                    if(point_added_flag == 1):
                        path_nodes[i].append(temp_node)
                        path_edge_weights[i].append(temp_weights)
        #print("---path m[x] - m[j] - m[k] - d[y],over---")
        # print(path_nodes)
        # m[x] - m[j] - d[k] - d[y]
        for j in range(miRNA_length):
            temp_path_nodes = []
            temp_path_edge_weights = []
            point_added_flag = 0  # judging the point added in paths or not
            # record initial point
            temp_path_nodes.extend([x])
            if (j == x):
                continue
            else:
                if (miRNA_miRNA_matrix[x][j] > threshold_value ):
                    # temporary save
                    temp_path_nodes.extend([j])  # record point m[j]
                    temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    if(k == y):
                        continue
                    elif (current_associated_A[j][k] == 1 and
                                disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([current_associated_A[j][k]])  # record weight  m[j] - d[k]
                        temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                    if(point_added_flag == 1):
                        path_nodes[i].append(temp_node)
                        path_edge_weights[i].append(temp_weights)
        #print("---path m[x] - m[j] - d[k] - d[y],over---")
        # print(path_nodes)
        # m[x] - d[j] - d[k] - d[y]
        for j in range(disease_length):
            temp_path_nodes = []
            temp_path_edge_weights = []
            point_added_flag = 0  # judging the point added in paths or not
            # record initial point
            temp_path_nodes.extend([x])
            if (j == y):
                continue
            else:
                if (current_associated_A[x][j] ==1 ):
                    # temporary save
                    temp_path_nodes.extend([600+j])  # record point d[j]
                    temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    if(k == j or k == y):
                        continue
                    elif (disease_disease_matrix[j][k] > threshold_value and
                                disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([disease_disease_matrix[j][k]])  # record weight  d[j] - d[k]
                        temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        path_nodes[i].append(temp_node)
                        path_edge_weights[i].append(temp_weights)
        #print("---path m[x] - d[j] - d[k] - d[y],over---")
        # m[x] - d[j] - m[k] - d[y]
        for j in range(disease_length):
                        temp_path_nodes = []
                        temp_path_edge_weights = []
                        # record initial point
                        temp_path_nodes.extend([x])
                        if (j == y):
                            continue
                        else:
                            if (current_associated_A[x][j] == 1):
                                # temporary save
                                temp_path_nodes.extend([600 + j])  # record point d[j]
                                temp_path_edge_weights.extend(
                                    [current_associated_A[x][j]])  # record weight  m[x] - d[j]
                                for k in range(miRNA_length):
                                    temp_node = copy.deepcopy(temp_path_nodes)
                                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                                    point_added_flag = 0  # judging the point added in paths or not
                                    if (k == x):
                                        continue
                                    if (current_associated_A[k][j] == 1 and
                                                current_associated_A[k][y] == 1):
                                        point_added_flag = 1
                                        # temporary save
                                        temp_node.extend([k])  # record point d[k]
                                        temp_node.extend([600 + y])  # record point d[y]
                                        temp_weights.extend([current_associated_A[k][j]])  # record weight  d[j] - m[k]
                                        temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                                        path_nodes.append(temp_node)
                                        path_edge_weights.append(temp_weights)

    #print(path_nodes[0])

    return path_nodes,path_edge_weights

#6 according to current Association Matrix, search points path  for single sample
def path_node1(current_data,current_associated_A,miRNA_miRNA_matrix,disease_disease_matrix,threshold_value):
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
    nodes_A = [x,y]
    value_A = [value]
    path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
    path_edge_weights.append(value_A)  # the first  value is A[x,y]

    # -----------first search sample i path length = 2-----------  #
    #print("---4.star search path---")
    # m[x] - m[j] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                temp_path_nodes.extend([600 + y])  # record point d[y]
                temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                # permanent save
                path_nodes.append(temp_path_nodes)
                path_edge_weights.append(temp_path_edge_weights)
    #print("---path m[x] - m[j] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                temp_path_nodes.extend([600 + y])  # record point d[y]
                temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                # permanent save
                path_nodes.append(temp_path_nodes)
                path_edge_weights.append(temp_path_edge_weights)
    #print("---path m[x] - d[j] - d[y],over---")
    # print(path_nodes)

    # -----------then search sample i path length = 3-----------  #
    # m[x] - m[j] - m[k] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        temp_path_edge_weights = []

        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(miRNA_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == x):
                        continue
                    elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                                current_associated_A[k][y] == 1):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([k])  # record point m[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([miRNA_miRNA_matrix[j][k]])  # record weight  m[j] - m[k]
                        temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        path_nodes.append(temp_node)
                        path_edge_weights.append(temp_weights)
    #print("---path m[x] - m[j] - m[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - m[j] - d[k] - d[y]
    for j in range(miRNA_length):
        temp_path_nodes = []
        temp_path_edge_weights = []

        # record initial point
        temp_path_nodes.extend([x])
        if (j == x):
            continue
        else:
            if (miRNA_miRNA_matrix[x][j] > threshold_value):
                # temporary save
                temp_path_nodes.extend([j])  # record point m[j]
                temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == y):
                        continue
                    elif (current_associated_A[j][k] == 1 and
                                disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([current_associated_A[j][k]])  # record weight  m[j] - d[k]
                        temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        path_nodes.append(temp_node)
                        path_edge_weights.append(temp_weights)
    #print("---path m[x] - m[j] - d[k] - d[y],over---")
    # print(path_nodes)
    # m[x] - d[j] - d[k] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if ( current_associated_A[x][j] == 1 ):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(disease_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if (k == j or k == y):
                        continue
                    if (disease_disease_matrix[j][k] > threshold_value and
                            disease_disease_matrix[k][y] > threshold_value):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([600 + k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([disease_disease_matrix[j][k]])  # record weight  d[j] - d[k]
                        temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                        path_nodes.append(temp_node)
                        path_edge_weights.append(temp_weights)
    # m[x] - d[j] - m[k] - d[y]
    for j in range(disease_length):
        temp_path_nodes = []
        temp_path_edge_weights = []
        # record initial point
        temp_path_nodes.extend([x])
        if (j == y):
            continue
        else:
            if (current_associated_A[x][j] == 1):
                # temporary save
                temp_path_nodes.extend([600 + j])  # record point d[j]
                temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                for k in range(miRNA_length):
                    temp_node = copy.deepcopy(temp_path_nodes)
                    temp_weights = copy.deepcopy(temp_path_edge_weights)
                    point_added_flag = 0  # judging the point added in paths or not
                    if ( k == x ):
                        continue
                    if (current_associated_A[k][j] == 1 and
                            current_associated_A[k][y] == 1):
                        point_added_flag = 1
                        # temporary save
                        temp_node.extend([k])  # record point d[k]
                        temp_node.extend([600 + y])  # record point d[y]
                        temp_weights.extend([current_associated_A[k][j]])  # record weight  d[j] - m[k]
                        temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                        path_nodes.append(temp_node)
                        path_edge_weights.append(temp_weights)
    #print("---looking path in current data, over")

    #print("---path m[x] - d[j] - d[k] - d[y],over---")    [374, 911, 758]

    return path_nodes,path_edge_weights

#7 save it's goal and previous value and paths for every sample in a certain data set
def save_paths_nodes(data_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value=0):  # just for one validation
    length_set = len(data_set)
    path_node_all = []
    path_edge_weights_all = []
    print("---4.star search path---")
    for i in range(length_set):
        temp_path_nodes, temp_path_edge_weights = path_node(data_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
        path_node_all.append(temp_path_nodes)
        path_edge_weights_all.append(temp_path_edge_weights)
        if(i%10==0):
            print("---This is %d",i)
    np.savez('nodes',path_node_all)
    np.savez('weights',path_edge_weights_all)


#7 save it's goal and previous value and paths for samples in train/dev/test set
def save_paths_nodes1(train_set,dev_set,test_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value=0):  # just for one validation
    print("---4.star search path---")
    # save train set
    length_train_set = len(train_set)
    path_node_train_set = []
    path_edge_weights_train_set = []
    for i in range(length_train_set):
        temp_path_nodes, temp_path_edge_weights = path_node(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
        path_node_train_set.append(temp_path_nodes)
        path_edge_weights_train_set.append(temp_path_edge_weights)
        if(i%100==0):
            print("---This is %d in train set",i)
        np.savez('./get_data/train_nodes', train_set=path_node_train_set)
        np.savez('./get_data/train_weights', train_set=path_edge_weights_train_set)
    # save dev set
    length_dev_set = len(dev_set)
    path_node_dev_set = []
    path_edge_weights_dev_set = []
    for i in range(length_dev_set):
        temp_path_nodes, temp_path_edge_weights = path_node(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
                                                            disease_disease_matrix, threshold_value)
        path_node_dev_set.append(temp_path_nodes)
        path_edge_weights_dev_set.append(temp_path_edge_weights)
        if (i % 50 == 0):
            print("---This is %d in dev set", i)
    np.savez('./get_data/dev_nodes',  dev_set=path_node_dev_set)
    np.savez('./get_data/dev_weights', dev_set=path_edge_weights_dev_set)
    # save test set
    length_test_set = len(test_set)
    path_node_test_set = []
    path_edge_weights_test_set = []
    for i in range(length_test_set):
        temp_path_nodes, temp_path_edge_weights = path_node(test_set[i], current_associated_A, miRNA_miRNA_matrix,
                                                            disease_disease_matrix, threshold_value)
        path_node_test_set.append(temp_path_nodes)
        path_edge_weights_test_set.append(temp_path_edge_weights)
        if (i % 100 == 0):
            print("---This is %d in test set", i)

    np.savez('./get_data/test_nodes',test_set = path_node_test_set)
    np.savez('./get_data/test_weights',test_set = path_edge_weights_test_set)

#7 every 500 samples  in train/dev/test set
def save_paths_nodes2(train_set,dev_set,test_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value=0):  # just for one validation
    print("---4.star search path---")
    # save train set
    length_train_set = len(train_set)
    path_node_train_set = []
    path_edge_weights_train_set = []
    for i in range(length_train_set):#7124
        temp_path_nodes, temp_path_edge_weights = path_node(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
        path_node_train_set.append(temp_path_nodes)
        path_edge_weights_train_set.append(temp_path_edge_weights)
        if(i == 7123):
            np.savez('./get_data/path_search/train/train_nodes' + str(i + 1), train_set=path_node_train_set)
            np.savez('./get_data/path_search/train/train_weights' + str(i + 1), train_set=path_edge_weights_train_set)
            path_node_train_set = []
            path_edge_weights_train_set = []
            print("---Train data path search over")
        if((i+1)%500==0):
            print("---This is %d in train set"%(i))
            np.savez('./get_data/path_search/train/train_nodes'+str(i+1), train_set=path_node_train_set)
            np.savez('./get_data/path_search/train/train_weights'+str(i+1), train_set=path_edge_weights_train_set)
            path_node_train_set = []
            path_edge_weights_train_set = []

    # save dev set
    length_dev_set = len(dev_set)
    path_node_dev_set = []
    path_edge_weights_dev_set = []
    for i in range(length_dev_set):#2036
        temp_path_nodes, temp_path_edge_weights = path_node(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
                                                            disease_disease_matrix, threshold_value)
        path_node_dev_set.append(temp_path_nodes)
        path_edge_weights_dev_set.append(temp_path_edge_weights)
        if (i == 2035):
            np.savez('./get_data/path_search/dev/dev_nodes' + str(i + 1), dev_set=path_node_dev_set)
            np.savez('./get_data/path_search/dev/dev_weights' + str(i + 1), dev_set=path_edge_weights_dev_set)
            path_node_dev_set = []
            path_edge_weights_dev_set = []
            print("---Train data path search over")
        if ((i + 1) % 500 == 0):
            print("---This is %d in dev set" % (i))
            np.savez('./get_data/path_search/dev/dev_nodes' + str(i+1), dev_set=path_node_dev_set)
            np.savez('./get_data/path_search/dev/dev_weights' + str(i+1), dev_set=path_edge_weights_dev_set)
            path_node_dev_set = []
            path_edge_weights_dev_set = []
    # save test set
    length_test_set = len(test_set)
    path_node_test_set = []
    path_edge_weights_test_set = []
    for i in range(length_test_set):#155160
        temp_path_nodes, temp_path_edge_weights = path_node(test_set[i], current_associated_A, miRNA_miRNA_matrix,
                                                            disease_disease_matrix, threshold_value)
        path_node_test_set.append(temp_path_nodes)
        path_edge_weights_test_set.append(temp_path_edge_weights)
        if (i == 155159):
            np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
            np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
            path_node_test_set = []
            path_edge_weights_test_set = []
            print("---Train data path search over")
        if ((i + 1) % 500 == 0):
            print("---This is %d in test set" % (i))
            np.savez('./get_data/path_search/test/test_nodes' + str(i + 1), test_set=path_node_test_set)
            np.savez('./get_data/path_search/test/test_weights' + str(i + 1), test_set=path_edge_weights_test_set)
            path_node_test_set = []
            path_edge_weights_test_set = []

#-------------------not save weights------------#

# 6 according to current Association Matrix, search points path  for single sample
def path_node2(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
            #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
            #  ------------------just using simple method, the index plus 600 -------------------  #
            # length

            miRNA_length = len(miRNA_miRNA_matrix)
            disease_length = len(disease_disease_matrix)

            """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
            path_nodes = []
            # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
            #path_edge_weights = []

            # for sample,from m[x] to d[y]
            # get index
            x = current_data.index_x  # miRNA
            y = current_data.index_y  # disease
            value = current_data.value
            # -----------for save in file-----------  #
            # -----------record m[x] - d[y] 's indedx and  value in matrix A-----------  #
            # -----------put index [x,y] in path_nodes,put the value in path_edge_weights-----------  #
            nodes_A = [x, y]
            #value_A = [value]
            path_nodes.append(nodes_A)  # the first is [x,y] , m[x] - d[y]
            #path_edge_weights.append(value_A)  # the first  value is A[x,y]

            # -----------first search sample i path length = 2-----------  #
            # print("---4.star search path---")
            # m[x] - m[j] - d[y]
            for j in range(miRNA_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []
                # record initial point
                temp_path_nodes.extend([x])
                if (j == x):
                    continue
                else:
                    if (miRNA_miRNA_matrix[x][j] > threshold_value and current_associated_A[j][y] == 1):
                        # temporary save
                        temp_path_nodes.extend([j])  # record point m[j]
                        temp_path_nodes.extend([600 + y])  # record point d[y]
                        #temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                        #temp_path_edge_weights.extend([current_associated_A[j][y]])  # record weight  m[j] - d[y]
                        # permanent save
                        path_nodes.append(temp_path_nodes)
                        #path_edge_weights.append(temp_path_edge_weights)
            # print("---path m[x] - m[j] - d[y],over---")
            # print(path_nodes)
            # m[x] - d[j] - d[y]
            for j in range(disease_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []
                # record initial point
                temp_path_nodes.extend([x])
                if (j == y):
                    continue
                else:
                    if (current_associated_A[x][j] == 1 and disease_disease_matrix[j][y] > threshold_value):
                        # temporary save
                        temp_path_nodes.extend([600 + j])  # record point d[j]
                        temp_path_nodes.extend([600 + y])  # record point d[y]
                        #temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                        #temp_path_edge_weights.extend([disease_disease_matrix[j][y]])  # record weight  d[j] - d[y]
                        # permanent save
                        path_nodes.append(temp_path_nodes)
                        #path_edge_weights.append(temp_path_edge_weights)
            # print("---path m[x] - d[j] - d[y],over---")
            # print(path_nodes)

            # -----------then search sample i path length = 3-----------  #
            # m[x] - m[j] - m[k] - d[y]
            for j in range(miRNA_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []

                # record initial point
                temp_path_nodes.extend([x])
                if (j == x):
                    continue
                else:
                    if (miRNA_miRNA_matrix[x][j] > threshold_value):
                        # temporary save
                        temp_path_nodes.extend([j])  # record point m[j]
                        #temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                        for k in range(miRNA_length):
                            temp_node = copy.deepcopy(temp_path_nodes)
                            #temp_weights = copy.deepcopy(temp_path_edge_weights)
                            point_added_flag = 0  # judging the point added in paths or not
                            if (k == j or k == x):
                                continue
                            elif (miRNA_miRNA_matrix[j][k] > threshold_value and
                                          current_associated_A[k][y] == 1):
                                point_added_flag = 1
                                # temporary save
                                temp_node.extend([k])  # record point m[k]
                                temp_node.extend([600 + y])  # record point d[y]
                                #temp_weights.extend([miRNA_miRNA_matrix[j][k]])  # record weight  m[j] - m[k]
                                #temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                                path_nodes.append(temp_node)
                                #path_edge_weights.append(temp_weights)
            # print("---path m[x] - m[j] - m[k] - d[y],over---")
            # print(path_nodes)
            # m[x] - m[j] - d[k] - d[y]
            for j in range(miRNA_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []

                # record initial point
                temp_path_nodes.extend([x])
                if (j == x):
                    continue
                else:
                    if (miRNA_miRNA_matrix[x][j] > threshold_value):
                        # temporary save
                        temp_path_nodes.extend([j])  # record point m[j]
                        #temp_path_edge_weights.extend([miRNA_miRNA_matrix[x][j]])  # record weight  m[x] - m[j]
                        for k in range(disease_length):
                            temp_node = copy.deepcopy(temp_path_nodes)
                            #temp_weights = copy.deepcopy(temp_path_edge_weights)
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
                                #temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                                path_nodes.append(temp_node)
                                #path_edge_weights.append(temp_weights)
            # print("---path m[x] - m[j] - d[k] - d[y],over---")
            # print(path_nodes)
            # m[x] - d[j] - d[k] - d[y]
            for j in range(disease_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []
                # record initial point
                temp_path_nodes.extend([x])
                if (j == y):
                    continue
                else:
                    if (current_associated_A[x][j] == 1):
                        # temporary save
                        temp_path_nodes.extend([600 + j])  # record point d[j]
                        #temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                        for k in range(disease_length):
                            temp_node = copy.deepcopy(temp_path_nodes)
                            #temp_weights = copy.deepcopy(temp_path_edge_weights)
                            point_added_flag = 0  # judging the point added in paths or not
                            if (k == j or k == y):
                                continue
                            if (disease_disease_matrix[j][k] > threshold_value and
                                        disease_disease_matrix[k][y] > threshold_value):
                                point_added_flag = 1
                                # temporary save
                                temp_node.extend([600 + k])  # record point d[k]
                                temp_node.extend([600 + y])  # record point d[y]
                                #temp_weights.extend([disease_disease_matrix[j][k]])  # record weight  d[j] - d[k]
                                #temp_weights.extend([disease_disease_matrix[k][y]])  # record weight  d[k] - d[y]
                                path_nodes.append(temp_node)
                                #path_edge_weights.append(temp_weights)
            # m[x] - d[j] - m[k] - d[y]
            for j in range(disease_length):
                temp_path_nodes = []
                #temp_path_edge_weights = []
                # record initial point
                temp_path_nodes.extend([x])
                if (j == y):
                    continue
                else:
                    if (current_associated_A[x][j] == 1):
                        # temporary save
                        temp_path_nodes.extend([600 + j])  # record point d[j]
                        #temp_path_edge_weights.extend([current_associated_A[x][j]])  # record weight  m[x] - d[j]
                        for k in range(miRNA_length):
                            temp_node = copy.deepcopy(temp_path_nodes)
                            #temp_weights = copy.deepcopy(temp_path_edge_weights)
                            point_added_flag = 0  # judging the point added in paths or not
                            if (k == x):
                                continue
                            if (current_associated_A[k][j] == 1 and
                                        current_associated_A[k][y] == 1):
                                point_added_flag = 1
                                # temporary save
                                temp_node.extend([k])  # record point d[k]
                                temp_node.extend([600 + y])  # record point d[y]
                                #temp_weights.extend([current_associated_A[k][j]])  # record weight  d[j] - m[k]
                                #temp_weights.extend([current_associated_A[k][y]])  # record weight  m[k] - d[y]
                                path_nodes.append(temp_node)
                                #path_edge_weights.append(temp_weights)
            # print("---looking path in current data, over")

            # print("---path m[x] - d[j] - d[k] - d[y],over---")    [374, 911, 758]

            return path_nodes


# 6 according to current Association Matrix, search points path  for single sample
def path_three_node(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
    #   -----for distinguishing the recorded node in path come from miRNA or disease -----  #
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

    return path_nodes  #
#######################------------------test
def path_node3(current_data, current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value):
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

#6 according to current Association Matrix, search points path  for all sample
def paths_nodes3(current_data_set,current_associated_A,SM,SD):
    #  -----for distinguishing the recorded node in path come from miRNA or disease -----  #
    #  ------------------just using simple method, the index plus 600 -------------------  #
    # length
    data_length = len(current_data_set)
    miRNA_length = len(SM)
    disease_length = len(SD)

    """ record path's nodes   [[sample1[path1 nodes]  [path2 nodes] ... [pathM nodes]]  
                               [sample2[path1 nodes]  [path2 nodes] ... [pathM nodes]]
                                ...
                               [sampleN[path1 nodes]  [path2 nodes] ... [pathM nodes]]]"""
    path_nodes = [[] for row in range(data_length)]

    # like path_nodes,record edges' weight [[sample1[path1 weights]  [path2 weights] ... [pathn weights]] ...]
    #path_edge_weights = [[] for row in range(data_length)]
    print("---4.star search path---")
    # for every sample,from m[x] to d[y]
    print("---two jump path---")
    # m[x] - m[j] - d[y]
    for i in range(miRNA_length):
        for j in range(miRNA_length):
            for k in range(disease_length):
                if(SM[i][j]==1 and current_associated_A[j][k]==1):
                    path_nodes[i].append([[i],[j],[k+600]])
    # m[x] - d[j] - d[y]
    for i in range(miRNA_length):
        for j in range(disease_length):
            for k in range(disease_length):
                if(current_associated_A[i][j]==1 and SD[j][k]==1):
                    path_nodes[i].append([[i],[j+600],[k+600]])
    # -----------then search sample i path length = 3-----------  #
    print("---three jump path---")
    # m[x] - m[j] - m[k] - d[y]
    for i in range(miRNA_length):
        for j in range(miRNA_length):
            for k in range(miRNA_length):
                for l in range(disease_length):
                    if(i!=k and SM[i][j]==1 and SM[j][k]==1 and current_associated_A[k][l]==1):
                        path_nodes[i].append([[i],[j],[k],[l+600]])
    print("---three jump path one over---")
    # m[x] - m[j] - d[k] - d[y]
    for i in range(miRNA_length):
        for j in range(miRNA_length):
            for k in range(disease_length):
                for l in range(disease_length):
                    if(SM[i][j]==1 and current_associated_A[j][k]==1 and SD[k][l]==1):
                        path_nodes[i].append([[i],[j],[k+600],[l+600]])
    print("---three jump path two over---")
    # m[x] - d[j] - d[k] - d[y]
    for i in range(miRNA_length):
        for j in range(disease_length):
            for k in range(disease_length):
                for l in range(disease_length):
                    if(j!=l and current_associated_A[i][j]==1 and SD[j][k]==1 and SD[k][l]==1):
                        path_nodes[i].append([[i],[j+600],[k+600],[l+600]])
    print("---three over")
    # m[x] - d[j] - m[k] - d[y]
    for i in range(miRNA_length):
        for j in range(disease_length):
            for k in range(miRNA_length):
                for l in range(disease_length):
                    if(i!=k and j!=l and current_associated_A[i][j]==1 and current_associated_A[j][k]==1 and current_associated_A[k][l]==1):
                        path_nodes[i].append([[i],[j+600],[k],[l+600]])



    return path_nodes

def make_matrix_SM_SD(SM,SD,threshold):
    SM_length  = len(SM)
    SD_length = len(SD)

    new_SM = [[0 for row in range(SM_length)]for row in range(SM_length)]
    #print(np.shape(new_SM))
    new_SD = [[0for row in range(SD_length)] for row in range(SD_length)]
    for i in range(SM_length):
        for j in range(SM_length):
            if(i!=j and SM[i][j]>threshold):
                new_SM[i][j] = 1
    for i in range(SD_length):
        for j in range(SD_length):
            if(i!=j and SD[i][j]>threshold):
                new_SD[i][j] = 1
    return new_SM,new_SD

#######################------------------test


#7 every 500 samples  in train/dev/test set
def save_paths_nodes3(train_set,dev_set,test_set,current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value=0):  # just for one validation
    print("---4.star search path---")
    # save train set
    length_train_set = len(train_set)
    path_node_train_set = []
    #path_edge_weights_train_set = []
    for i in range(length_train_set):#7124
        temp_path_nodes = path_node2(train_set[i], current_associated_A, miRNA_miRNA_matrix, disease_disease_matrix, threshold_value)
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
        temp_path_nodes = path_node2(dev_set[i], current_associated_A, miRNA_miRNA_matrix,
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
        temp_path_nodes = path_node2(test_set[i], current_associated_A, miRNA_miRNA_matrix,
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

#[sample[[goal][paht2][path3]]]

if __name__ == '__main__':
    # define parameters
    #k = 10
    #all_k_train_set, all_k_development_set, all_k_test_set, all_k_count_positive, A, num, A_length, sava_association_A, Sm, Sd,save_all_count_A = dp.get_pre_train_dev_test_set(k)
    #path_nodes, path_edge_weights = path_node(all_k_train_set[0][0],save_all_count_A[0],Sm,Sd,0)
    #path_nodes, path_edge_weights = paths_nodes(all_k_train_set[0], save_all_count_A[0], Sm, Sd, 0)
    #print(path_nodes[-2])
    #print(path_edge_weights[0])
    A, Sm, Sd = dp.read_data_flies()
    new_SM ,new_SD  = make_matrix_SM_SD(Sm,Sd,0.2)# if bigger than threshold
    print(np.shape(new_SM))
    print(np.shape(new_SD))

    D = np.load('./get_data/data_prepare/devided_data.npz')
    all_k_train_set = D['all_k_train_set']
    all_k_development_set = D['all_k_development_set']
    all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']

    path_ = paths_nodes3(all_k_train_set[0], save_all_count_A[0], new_SM, new_SD)
    print(len(path_))
    print(np.shape(path_))
    """
    A, Sm, Sd = dp.read_data_flies()
    D = np.load('./get_data/data_prepare/devided_data.npz')
    all_k_train_set = D['all_k_train_set']
    all_k_development_set = D['all_k_development_set']
    all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']

    save_paths_nodes3(all_k_train_set[0],all_k_development_set[0],all_k_test_set[0],save_all_count_A[0],Sm,Sd, threshold_value=0)
    #for i in range(1000):
    #path_nodes, path_edge_weights = path_node(all_k_train_set[0][i], save_all_count_A[0], Sm, Sd, 0)
    #print(path_nodes[-1])
"""