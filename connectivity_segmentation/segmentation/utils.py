import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from connectivity_segmentation.metrics.scores import get_nbr_microstates


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def smooth_segm(labels, n_win, nbr_clust):
    """
    Median filter
    :param labels:
    :param n_win:
    :param nbr_clust:
    :return:
    """
    for w in n_win:
        labels_copy = np.copy(labels)
        for l in range(w, len(labels) - w):
            s = np.array([np.sum(labels[l - w : l + w] == j) for j in range(nbr_clust)])
            labels_copy[l] = np.argmax(s)
        nbr_microstates = get_nbr_microstates(labels_copy)
        if nbr_microstates == nbr_clust:
            break
    if len(n_win) > 1 and nbr_microstates != nbr_clust:
        raise ValueError(f"The 'strict' smoothing could not produce '{k}' distinct clusters")
    return labels_copy
