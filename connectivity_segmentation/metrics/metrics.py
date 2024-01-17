import numpy as np
import random
from numpy import ndarray
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


# def get_centroids_stats(centroids):
#     centroids_val = np.zeros((len(centroids.shape[-1])))
#     for k in range(len(centroids.shape[-1])):
#         centroids_val[k] = round(np.sqrt(np.sum(centroids[:, :, k] ** 2)), 3)
#     return centroids_val


def _init_centroids_idx(data, K, min_time=30):
    return random.sample(range(0, data.shape[2], min_time), K).sort()


def compute_gcp(data, takeMST, isSymetrical):
    """
    Global connectivity power (global field power) is the spatial standard deviation of each sample to the sample
    mean. Sum of each sample deviation to G(t) mean then equal to the total std of samples to mean.
    :return:
    """
    # if takeMST:
    #     gcp = np.std(np.reshape(data, (n_ch * n_ch, n_t)), axis=0)
    # else:
    gcp = np.zeros(data.shape[-1])
    if takeMST:  # take only non-zeros values
        for t in range(data.shape[-1]):
            nonzer_val = []
            tot_nonzer_val = 0
            for i in range(data.shape[0]):
                for j in range(data.shape[0]):
                    if data[i, j, t]:
                        if isSymetrical:
                            nonzer_val.append(data[i, j, t])
                        else:
                            nonzer_val.append(
                                data[i, j, t] / 2
                            )  # Because each values are here 2 times
                        tot_nonzer_val += 1
            nonzer_val = np.array(nonzer_val)
            avg_val = np.mean(nonzer_val)
            if not isSymetrical:
                tot_nonzer_val /= 2
            gcp[t] = np.square(np.sum((nonzer_val - avg_val) ** 2) / tot_nonzer_val)
    else:
        for t in range(data.shape[-1]):
            if isSymetrical:
                gcp[t] = np.square(np.sum(data[:, :, t] ** 2) / data.shape[0])
            else:
                gcp[t] = np.square(
                    np.sum(data[:, :, t] ** 2) / (data.shape[0] * (data.shape[0] - 1) / 2)
                )
    return gcp


def compute_pairwise_dist(data):
    dist_arr = np.zeros((data.shape[-1], data.shape[-1]))
    for sample_i in range(data.shape[-1]):
        for sample_j in range(data.shape[-1]):
            sc = np.sum(data[:, :, sample_i] * data[:, :, sample_j]) / (
                np.sqrt(np.sum(data[:, :, sample_i] ** 2))
                * np.sqrt(np.sum(data[:, :, sample_j] ** 2))
            )
            dist_arr[sample_i, sample_j] = abs(sc - 1)
    return dist_arr


def find_closest_val(list):
    temp = abs(max(list))
    for i in range(1, len(list)):
        num1 = list[i]
        j = i - 1
        while j >= 0:
            num2 = list[j]
            res = abs(num1 - num2)
            if res < temp:
                temp = res
                first_num = num1
                sec_num = num2
            j = j - 1
    return [first_num, sec_num]


def find_opt_k(cv_list):
    if len(cv_list) > 1:
        k_opt: ndarray[int] = np.argmax(find_closest_val(cv_list))
    else:
        k_opt = np.argmax(cv_list)
    return k_opt
