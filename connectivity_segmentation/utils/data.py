import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import networkx as nx


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def extract_data(file, list_files, stackFiles):
    if stackFiles:
        concat = []
        for dyn_con in list_files:
            with open(dyn_con, "rb") as file1:
                dyn_conn_matrix = pickle.load(file1)
            if not isinstance(concat, np.ndarray):
                concat = dyn_conn_matrix
            else:
                concat = np.concatenate((concat, dyn_conn_matrix), -1)
        n_ch = dyn_conn_matrix.shape[0]
        n_t = dyn_conn_matrix.shape[-1]
        data = concat
    else:
        with open(file, "rb") as file1:
            data = pickle.load(file1)
            print(file)
        n_ch = data.shape[0]
        n_t = data.shape[-1]
    return data, n_ch, n_t


def rearrange_data(data, n_t, list_files, takeMST, isSymetrical, useRefAvg):
    if takeMST:
        for t in range(data.shape[2]):
            G = nx.from_numpy_array(data[:, :, t])
            Max_tree = nx.maximum_spanning_tree(G)
            max_tree_con = nx.to_numpy_array(Max_tree)
            data[:, :, t] = max_tree_con
    # make the conn matrix asymmetric
    if not isSymetrical:
        for k in range(data.shape[0]):
            for l in range(data.shape[1]):
                if l >= k:
                    data[k, l, :] = 0
    if useRefAvg:
        for i in range(len(list_files)):
            for t in range(n_t):
                data[:, :, t] -= mean_adj_matrix(data[:, :, t], isSymetrical)
    return data
