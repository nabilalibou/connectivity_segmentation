import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def get_nbr_microstates(labels):
    nbr_microstates = 0
    microstate_val = None
    for idx in range(labels.shape[0]):
        if labels[idx] != microstate_val:
            microstate_val = labels[idx]
            nbr_microstates += 1
    return nbr_microstates


def count_labels_occurrence(labels, nbr_clust):
    label_stats = {"nbr_microstates": get_nbr_microstates(labels)}
    for k in range(nbr_clust):
        label_stats[f"Label {k + 1} density (%)"] = round(
            np.count_nonzero(labels == k) * 100 / labels.shape[0], 3
        )
    return label_stats


def label_count_stats(labels, clust_val, K_range):
    label_stats = {"nbr_microstates": get_nbr_microstates(labels)}
    for k in range(K_range[-1]):
        if k + 1 > clust_val:
            label_stats[f"Label {k + 1} density (%)"] = None
        else:
            label_stats[f"Label {k + 1} density (%)"] = round(
                np.count_nonzero(labels == k) * 100 / labels.shape[0], 3
            )
    return label_stats


def duration_labels_occurrence(labels, nbr_clust):
    label_stats = {}
    duration = []
    start = []
    stop = []
    for k in range(nbr_clust):
        for idx in range(labels.shape[0]):
            if labels[idx] == k:
                if labels[idx] == labels[idx + 1]:
                    if len(start) < k + 1:
                        start.append(idx)
                    if idx == labels.shape[0] - 2:
                        stop.append(idx + 1)
                        duration.append(stop[k] - start[k])
                        break
                else:
                    if len(stop) < k + 1:
                        stop.append(idx)
                        duration.append(stop[k] - start[k])
        label_stats[f"Label {k + 1} start/duration/stop"] = f"{start[k]}/{duration[k]}/{stop[k]}"
    return label_stats


def label_duration_stats(labels, clust_val, K_range):
    label_stats = {}
    duration = []
    start = []
    stop = []
    for k in range(K_range[-1]):
        if k + 1 > clust_val:
            label_stats[f"Label {k + 1} start/duration/stop"] = None
        else:
            for idx in range(labels.shape[0]):
                if labels[idx] == k:
                    if labels[idx] == labels[idx + 1]:
                        if len(start) < k + 1:
                            start.append(idx)
                        if idx == labels.shape[0] - 2:
                            stop.append(idx + 1)
                            duration.append(stop[k] - start[k])
                            break
                    else:
                        if len(stop) < k + 1:
                            stop.append(idx)
                            duration.append(stop[k] - start[k])
            label_stats[
                f"Label {k + 1} start/duration/stop"
            ] = f"{start[k]}/{duration[k]}/{stop[k]}"
    return label_stats
