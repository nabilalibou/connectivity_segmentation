import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def _distance_matrix(X, Y=None):
    """Distance matrix used in metrics."""
    distances = np.abs(1 / np.corrcoef(X, Y)) - 1
    distances = np.nan_to_num(distances, copy=False, nan=10e300, posinf=1e300, neginf=-1e300)
    return distances


def mean_adj_matrix(adj_matrix, isSymetrical=True):
    if isSymetrical:
        return np.mean(adj_matrix)
    else:
        return np.sum(adj_matrix) / (adj_matrix.shape[0] * (adj_matrix.shape[0] - 1) / 2)


def compute_inertia(data, template_graphs):
    return np.sum((data - template_graphs) ** 2)


def _fill_template_graphs(data, centroids, labels):
    """
    dot_product[:, :, t] = np.dot(template_graphs[:, :, t], data[:, :, t])  # Same trend in the results
    :param run:
    :return:
    """
    template_graphs = np.zeros((data.shape[0], data.shape[0], data.shape[2]))
    for t in range(data.shape[2]):
        template_graphs[:, :, t] = centroids[:, :, labels[t]]
        # dot_product[:, :, t] = np.dot(template_graphs[:, :, t], data[:, :, t])  # Same trend in the results
    return template_graphs


def compute_cross_val(data, K, centroids, labels):
    """
    Cross-Validation criterion to choose optimal number of clusters
    :return:
    """
    template_graphs = _fill_template_graphs(data, centroids, labels)
    var = np.sum(data**2) - np.sum(np.sum(template_graphs * data, axis=2) ** 2)
    # var = np.sum(data ** 2) - np.sum(np.sum(dot_product, axis=2) ** 2)  # try with dot product
    var /= data.shape[-1] * (data.shape[0] - 1)
    cv = var * (data.shape[0] - 1) ** 2 / (data.shape[0] - K - 1.0) ** 2
    return cv


def corr_conn(X, Y, diff_val=1e-3):
    # Compute spatial correlation between X and Y
    # check shape of X and Y <=
    sumXY = 0
    sumXX = 0
    sumYY = 0
    num = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] - Y[i, j] > diff_val:
                num += 1
                sumXY += X[i, j] * Y[i, j]
                sumXX += X[i, j] * X[i, j]
                sumYY += Y[i, j] * Y[i, j]
    if sumXX or sumYY:
        sc = sumXY / (np.sqrt(sumXX) * np.sqrt(sumYY))
    else:
        sc = 1.0
    # print(num)
    return sc
