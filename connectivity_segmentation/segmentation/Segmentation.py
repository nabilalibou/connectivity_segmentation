"""Segmentation class"""
import pickle
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Union
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from connectivity_segmentation.metrics.scores import duration_labels_occurrence, count_labels_occurrence
from utils import smooth_segm
from viz import plot_segmentation
from connectivity_segmentation.utils.result import save_param_csv
from connectivity_segmentation.utils.checks import _check_type


class Segmentation:
    """
    Class for a connectivity clusters segmentation.
    Adaptation of the _BaseSegmentation class from the library pycrostate [1] (https://github.com/vferat/pycrostates,
    Copyright (c) 2020, Victor Férat, All rights reserved.)
    References
    ----------
       [1] Victor Férat, Mathieu Scheltienne, rkobler, AJQuinn, & Lou. (2023).
           vferat/pycrostates: 0.4.1 (0.4.1).
           Zenodo. https://doi.org/10.5281/zenodo.10176055
    """
    def __init__(
        self,
        labels: NDArray[int],
        data: NDArray,
        run_opt: int,
        cluster_centers_: NDArray[float],
        cluster_names: Optional[List[str]] = None,
        predict_parameters: Optional[dict] = None,
    ):
        # check input
        _check_type(labels, (np.ndarray,), "labels")
        _check_type(data, (np.ndarray,), "data")
        _check_type(cluster_centers_, (np.ndarray,), "cluster_centers_")
        if cluster_centers_.ndim != 3:
            raise ValueError(
                "Argument 'cluster_centers_' should be a 3D array. The "
                f"provided array shape is {cluster_centers_.shape} which has "
                f"{cluster_centers_.ndim} dimensions."
            )
        self._data = data
        self._run_opt = self._check_run_opt(run_opt)
        self._cluster_centers = cluster_centers_
        self._cluster_names = self._check_cluster_names(cluster_names, self._cluster_centers)
        self._predict_parameters = self._check_predict_parameters(predict_parameters)
        # compute_parameters variables
        self._labels = labels
        self._run_labels_ = labels
        self._n_ch_ = data.shape[0]
        self._n_t_ = data.shape[-1]
        self._n_clust = cluster_centers_.shape[0]

    def plot(
        self,
        gcp,
        times,
        list_clusters,
    ):
        plot_segmentation(
            self._labels,
            gcp,
            times,
            list_clusters,
            cluster_names=self._cluster_centers,
            axes=None,
            cbar_axes=None,
            show=True,
            block=False,
        )

    def compute_parameters(self, nbr_clust, gev, tot_n_iter, smoothing, n_runs=1):
        check_consistency = True
        mean_similarity = np.zeros(n_runs - 1)
        dist_arr = np.zeros((self._n_t_, self._n_t_))
        params_names = [
            "total iterations",
            "GEV",
            "Silhouette",
            "Variance Ratio Criterion",
            "Davies-Bouldin",
        ]
        X = np.transpose(np.reshape(self._data, (self._n_ch_ * self._n_ch_, self._n_t_)))
        _labels = self._labels
        if len(np.unique(_labels)) != nbr_clust:
            raise ValueError(
                f"The partition does not contain all the clusters ({nbr_clust} clusters)"
            )

        if smoothing == "strict":
            for j in range(nbr_clust):
                params_names.insert(2 + j, f"Label {j + 1} start/duration/stop")
        else:
            params_names.insert(2, "nbr_microstates")
            for j in range(nbr_clust):
                params_names.insert(3 + j, f"Label {j + 1} density (%)")

        if check_consistency and n_runs > 1:
            params_names.append("Consistency")

        scores = {
            "Silhouette": None,
            "Variance Ratio Criterion": None,
            "Davies-Bouldin": None,
        }
        if check_consistency and n_runs > 1:
            scores["Consistency between runs"] = None
        print("Silhouette scores (with precomputed distances)")

        # Silhouette score
        # _distance_matrix(data)

        # dist_arr = np.zeros((sc_k.shape[0], sc_k.shape[0]))
        # for sample_i in range(sc_k.shape[0]):
        #     for sample_j in range(sc_k.shape[0]):
        #         dist_arr[sample_i, sample_j] = abs(retain_sc_k[i, sample_i] - retain_sc_k[i, sample_j])

        if smoothing:
            try:
                if smoothing < 0:
                    raise ValueError(
                        "smoothing value need to be eiter an integer > 0 or equal to 'strict'"
                    )
                else:
                    n_win = [smoothing]
            except TypeError:
                if smoothing != "strict":
                    raise ValueError(
                        "smoothing value need to be eiter an integer > 0 or equal to 'strict'"
                    )
                else:
                    n_win = [i for i in range(2, self._n_t_)]
            L = smooth_segm(_labels, n_win)
            _labels = L

        try:
            # silh_scores = silhouette_score(dist_arr, _labels, metric="precomputed")
            scores["Silhouette"] = silhouette_score(X, _labels)
            print(f"K-means {nbr_clust} clusters: {scores['Silhouette']}")
            if scores["Silhouette"] > 0.8:
                print("good clustering")
            elif scores["Silhouette"] > 0.5:
                print("acceptable and shows clusters are not overlapping")
            elif scores["Silhouette"] == 0:
                print("clusters are overlapping and either the data or the value of k is incorrect")
            elif scores["Silhouette"] < 0:
                print("elements have likely been assigned to the wrong clusters")
        except ValueError:
            print(
                "silhouette score could not be computed as there is only one label in the segmentation"
            )
        # Higher values = higher cluster density, better separation
        scores["Variance Ratio Criterion"] = calinski_harabasz_score(X, _labels)
        scores["Davies-Bouldin"] = davies_bouldin_score(
            X, _labels
        )  # lower values indicating better clustering

        # Check labels consistency between runs for each K
        if check_consistency and n_runs > 1:
            for r in range(n_runs - 1):
                mean_similarity[r] = np.mean(
                    (self._run_labels_[r] == self._run_labels_[r + 1]).astype(int)
                )
            scores["Consistency between runs"] = np.mean(mean_similarity).round(3)
            tot_similarity = round(np.mean(mean_similarity[r]) * 100, 3)  # mean is right ?
            # plt.plot(range(0, len(mean_similarity[i])), mean_similarity[i])
            # plt.show()
            tot_simi_dict = {f"K={nbr_clust}": tot_similarity}
            # print(f"Consistency index for each K (in %): \n{tot_simi_dict}")

        # Add statistics
        params_list = [tot_n_iter, gev.round(3)]

        # do a loop to extract all the keys and values automaticaly (do a loop + if cond)
        if smoothing == "strict":
            for j in range(nbr_clust):
                params_list.append(
                    duration_labels_occurrence(_labels, nbr_clust)[
                        f"Label {j + 1} start/duration/stop"
                    ]
                )
        else:
            count_stats = count_labels_occurrence(_labels, nbr_clust)
            params_list.append(count_stats["nbr_microstates"])
            for j in range(nbr_clust):
                params_list.append(count_stats[f"Label {j + 1} density (%)"])
        # params_list.append(np.mean(all_gev[self._run_opt, :]).round(3))
        # params_list.append(min(j for j in all_gev[self._run_opt, :] if j > 0).round(3))
        # params_list.append(np.max(all_gev[self._run_opt, :]).round(3))
        # params_list.append(
        #     (
        #         np.max(all_gev[self._run_opt, :])
        #         - min(j for j in all_gev[self._run_opt, :] if j > 0)
        #     ).round(3)
        # )
        params_list.append(round(scores["Silhouette"], 3))
        params_list.append(round(scores["Variance Ratio Criterion"], 3))
        params_list.append(round(scores["Davies-Bouldin"], 3))
        if check_consistency and n_runs > 1:
            params_list.append(scores["Consistency between runs"])

        return params_list, params_names

    def save_param(self, params_values, params_names, stats_path):
        save_param_csv(params_values, params_names, [self._n_clust], stats_path)

    def save_labels(self, labels_path):
        with open(labels_path, "wb") as f1:
            pickle.dump(self._labels, f1)
            print(f"file {labels_path} saved")

    # --------------------------------------------------------------------
    @staticmethod
    def _check_cluster_names(
        cluster_names: List[str],
        cluster_centers_: NDArray[float],
    ):
        """Check that the argument 'cluster_names' is valid."""
        _check_type(cluster_names, (list, None), "cluster_names")
        if cluster_names is None:
            return [str(k) for k in range(1, len(cluster_centers_) + 1)]
        else:
            if cluster_centers_.shape[-1] == len(cluster_names):
                return cluster_names
            else:
                raise ValueError(
                    "The same number of cluster centers and cluster names "
                    f"should be provided. There are {len(cluster_centers_)} "
                    f"cluster centers and '{len(cluster_names)}' provided."
                )

    @staticmethod
    def _check_predict_parameters(predict_parameters: dict):
        """Check that the argument 'predict_parameters' is valid."""
        _check_type(predict_parameters, (dict, None), "predict_parameters")
        if predict_parameters is None:
            return None
        valid_keys = (
            "factor",
            "tol",
            "half_window_size",
            "min_segment_length",
            "reject_edges",
            "reject_by_annotation",
        )
        # Let the door open for custom prediction with different keys, so log
        # a warning instead of raising.
        for key in predict_parameters.keys():
            if key not in valid_keys:
                print(
                    "The key '%s' in predict_parameters is not part of "
                    "the default set of keys supported by NKY-segmentation.",
                    key,
                )
        return predict_parameters

    @staticmethod
    def _check_run_opt(run_opt: int) -> int:
        """Check that run_opt is a non-negative integer."""
        _check_type(run_opt, ("int",), item_name="run_opt")
        if run_opt < 0:
            raise ValueError(
                f"Optimal run index must be an non-negative integer. Provided: '{run_opt}'."
            )
        return run_opt

    # --------------------------------------------------------------------

    @property
    def data(self) -> NDArray[float]:
        """Dynamic connectivity matrices (n_channels, n_channels, n_samples)

        :type: `~numpy.array`
        """
        return self._data.copy()

    @property
    def predict_parameters(self) -> dict:
        """Parameters used to predict the current segmentation.

        :type: `dict`
        """
        if self._predict_parameters is None:
            print("Argument 'predict_parameters' was not provided when creating the segmentation.")
        return self._predict_parameters.copy()

    @property
    def labels(self) -> NDArray[int]:
        """Microstate label attributed to each sample (the segmentation).

        :type: `~numpy.array`
        """
        return self._labels.copy()

    @property
    def _run_labels(self) -> NDArray[int]:
        """Microstate label attributed to each sample (the segmentation) by runs.

        :type: `~numpy.array`
        """
        return self._run_labels_.copy()

    @property
    def cluster_centers_(self) -> NDArray[float]:
        """Cluster centers (i.e topographies)
        used to compute the segmentation.

        :type: `~numpy.array`
        """
        return self._cluster_centers.copy()

    @property
    def cluster_names(self) -> List[str]:
        """Name of the cluster centers.

        :type: `list`
        """
        return self._cluster_names.copy()

    @property
    def n_clust(self) -> int:
        """Number of clusters (number of microstates).

        :type: `int`
        """
        return self._n_clust

    @property
    def _n_t(self) -> int:
        """number of time step from dynamic connectivity matrices.

        :type: `int`
        """
        return self._n_t_

    @property
    def _n_ch(self) -> int:
        """number of channels from dynamic connectivity matrices.

        :type: `int`
        """
        return self._n_ch_

    def run_opt(self) -> int:
        """Number corresponding to the optimal run

        :type: `int`
        """
        return self._run_opt
