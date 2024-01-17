import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def compute_corr(A, B):
    """
    Compute spatial correlation between matrices A and B
    :param A:
    :param B:
    :return:
    """
    return np.sum(A * B) / (np.sqrt(np.sum(A**2)) * np.sqrt(np.sum(B**2)))
