"""Class and functions to use modified Kmeans for tracking the dynamics of brain functional connectivity at millisecond
scale. See Mheich, A.;2015 doi:10.1016/j.jneumeth.2015.01.002 and Pascual-Marqui, 1995 doi:10.1109/10.391164."""
import pickle
from typing import List, Union
from itertools import groupby
import numpy as np
import random
from numpy.typing import NDArray
from scipy.signal import convolve2d
from connectivity_segmentation.segmentation.Segmentation import Segmentation
from utils import compute_corr
from connectivity_segmentation.metrics.scores import get_nbr_microstates
from connectivity_segmentation.utils.checks import _check_type


class ModKmeans:
    """
    Implementation of the modified Kmeans algorithm to cluster dynamic connectivity matrices.
    See Pascual-Marqui; 1995 doi:10.1109/10.391164.
    """

    def __init__(
        self, n_clust: int, n_runs: int = 1, max_iter: int = 50, tol: Union[int, float] = 1e-6
    ):
        self._n_clust = self._check_n_clusters(n_clust)
        self._cluster_names = [str(k) for k in range(self._n_clust)]
        self._max_iter = self._check_max_iter(max_iter)
        self._n_runs = self._check_n_runs(n_runs)
        self._tol = self._check_tol(tol)

        # fit variables
        self._n_t_ = None
        self._n_ch_ = None
        self._converg_iter = np.zeros(self._n_runs, dtype=int)
        self._gev_max = np.zeros(self._n_runs)
        self._gev_ = np.zeros((self._n_runs, self._max_iter))
        self._run_labels_ = None
        self._centroids = None
        self._all_centroids_by_run_ = None
        self._best_run = None
        self._fitted_data = None
        self._fitted = False

    def init_centroids_idx(self, min_time):
        return sorted(random.sample(range(0, self._n_t_, min_time), self._n_clust))

    def compute_corr_mat(self, data):
        """
        Compute spatial correlation between Gk and G(t)
        :param data:
        :param K:
        :return:
        """
        sc_k = np.zeros((self._n_t_, self._n_clust))
        for clust, k in enumerate(range(self._n_clust)):
            Gk = self._centroids[:, :, clust]
            for t in range(data.shape[2]):
                G = data[:, :, t]
                sc_k[t, clust] = compute_corr(Gk, G)
        return sc_k

    def _assign_labels(
        self,
        data: NDArray[float],
    ) -> NDArray[int]:
        sc_k = self.compute_corr_mat(data)
        labels = np.argmax(sc_k, axis=1)
        return labels

    def _update_centroids(self, data, labels):
        for clust in range(self._n_t_):
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    graph_clustered = np.where(labels == clust)[0]
                    if graph_clustered.tolist():
                        self._centroids[x, y, clust] = np.mean(data[x, y, graph_clustered])

    @staticmethod
    def _compute_gev(sc_k, labels):
        """
        Global explained variance (GEV) is calculated
        :param sc_k:
        :param labels:
        :return:
        """
        gev = np.zeros(sc_k.shape[1])
        for k in range(sc_k.shape[1]):
            k_sc_idx = np.where(labels == k)[0]
            gev[k] = np.sum(np.square(sc_k[k_sc_idx, :]))
        return np.sum(gev)

    def _find_closest_cluster(self, sc_k, labels, run):
        gev_total = self._compute_gev(sc_k, labels)
        if gev_total > self._gev_max[run]:
            self._gev_max[run] = gev_total
            self._run_labels_[run, :] = labels
            self._all_centroids_by_run_[run] = self._centroids
        return gev_total

    def _check_fit(self):
        """Check if the cluster is fitted."""
        if not self.fitted:
            raise RuntimeError(
                "Clustering algorithm must be fitted before using " f"{self.__class__.__name__}"
            )
        # sanity-check
        assert self._centroids is not None
        assert self._fitted_data is not None
        assert self._run_labels_ is not None

    def _check_unfitted(self):
        """Check if the cluster is unfitted."""
        if self.fitted:
            raise RuntimeError(
                "Clustering algorithm must be unfitted before using "
                f"{self.__class__.__name__}. You can set the property "
                "'.fitted' to False if you want to remove the instance fit."
            )
        # sanity-check
        assert self._centroids is None
        assert self._fitted_data is None
        assert self._run_labels_ is None

    def fit(self, data, min_time=30):
        self._n_ch_ = data.shape[0]
        self._n_t_ = data.shape[-1]
        self._run_labels_ = np.zeros((self._n_runs, self._n_t_), dtype=int)
        var_arr = np.zeros((self._n_runs, self._max_iter))
        self._all_centroids_by_run_ = np.zeros(
            (self._n_runs, data.shape[0], data.shape[1], self._n_clust)
        )
        for run in range(self._n_runs):
            # Initialization: randomly pick Gk graphs from G(t)
            self._centroids = np.zeros((data.shape[0], data.shape[1], self._n_clust))
            K_init_idx = self.init_centroids_idx(min_time)
            for clust in range(self._n_clust):
                self._centroids[:, :, clust] = data[:, :, K_init_idx[clust]]
            n_iter = 0
            var0 = 1.0
            var1 = 0.0
            # convergence criterion: variance estimate
            while (np.abs((var0 - var1) / var0) > self._tol) & (n_iter < self._max_iter):
                sc_k = self.compute_corr_mat(data)
                labels = np.argmax(sc_k, axis=1)
                self._update_centroids(data, labels)
                self._gev_[run, n_iter] = self._find_closest_cluster(sc_k, labels, run)
                var1 = var0
                var0 = self._gev_[run, n_iter]
                var_arr[run, n_iter] = np.abs((var0 - var1) / var0)
                n_iter += 1
            if n_iter < self._max_iter:
                print(f"\trun {run+1}/{self._n_runs} converged after {n_iter:d} iterations.")
            else:
                print(
                    f"\trun {run+1}/{self._n_runs} did NOT converge after {self._max_iter} iterations."
                )
            self._converg_iter[run] = n_iter
        self._best_run = np.argmax(self._gev_max)
        self._centroids = self._all_centroids_by_run_[self._best_run]
        self._fitted_data = data
        self._fitted = True

    def predict(
        self,
        data,
        factor: int = 0,
        half_window_size: int = 1,
        tol: Union[int, float] = 10e-6,
        min_segment_length: int = 0,
        reject_edges: bool = False,
    ):
        predict_parameters = {
            "factor": factor,
            "tol": tol,
            "half_window_size": half_window_size,
            "min_segment_length": min_segment_length,
            "reject_edges": reject_edges,
        }
        if factor == 0:
            print("Segmenting data without smoothing.")
        else:
            print(
                "Segmenting data with factor %s and effective smoothing " "window size: %.4f (s).",
                factor,
                (2 * half_window_size + 1) / data.shape[-1],
            )
        if min_segment_length > 0:
            print(
                "Rejecting segments shorter than %.4f (ms).",
                min_segment_length / data.shape[-1],
            )
        if reject_edges:
            print("Rejecting first and last segments.")

        labels = self._assign_labels(data)
        # labels = ModKmeans._median_smooth(labels, [half_window_size], self._n_clust)
        # if factor != 0:
        #     labels = self._smooth_segmentation(
        #         data, self._centroids, labels, factor, tol, half_window_size
        #     )
        if reject_edges:
            labels = self._reject_edge_segments(labels)

        if 0 < min_segment_length:
            labels = self._reject_short_segments(labels, data, min_segment_length)

        # Provide properties to copy the arrays
        return Segmentation(
            labels=labels,
            data=data,
            run_opt=self._best_run,
            cluster_centers_=self._centroids,
            cluster_names=self._cluster_names,
            predict_parameters=predict_parameters,
        )

    @staticmethod
    def _smooth_segmentation(
        data: NDArray[float],
        states: NDArray[float],
        labels: NDArray[int],
        factor: int,
        tol: Union[int, float],
        half_window_size: int,
    ) -> NDArray[int]:
        """Apply smoothing.

        Algorithm taken from the library pycrostate [1] (https://github.com/vferat/pycrostates, Copyright (c) 2020,
        Victor Férat, All rights reserved.) which is adapted from [2].

        factor : int
            Factor used for label smoothing. ``0`` means no smoothing. Default to 0.
        half_window_size : int
            Number of samples used for the half window size while smoothing labels. The
            half window size is defined as ``window_size = 2 * half_window_size + 1``.
            It has no effect if ``factor=0`` (default). Default to 1.
        tol : float
            Convergence tolerance.
        min_segment_length : int
            Minimum segment length (in samples). If a segment is shorter than this
            value, it will be recursively reasigned to neighbouring segments based on
            absolute spatial correlation.
        reject_edges : bool
            If ``True``, set first and last segments to unlabeled.

        References
        ----------
           [1] Victor Férat, Mathieu Scheltienne, rkobler, AJQuinn, & Lou. (2023).
               vferat/pycrostates: 0.4.1 (0.4.1).
               Zenodo. https://doi.org/10.5281/zenodo.10176055
           [2] R. D. Pascual-Marqui, C. M. Michel and D. Lehmann.
               Segmentation of brain electrical activity into microstates:
               model estimation and validation.
               IEEE Transactions on Biomedical Engineering,
               vol. 42, no. 7, pp. 658-665, July 1995,
               https://doi.org/10.1109/10.391164.
        """
        Ne, Nt = data.shape
        Nu = states.shape[0]
        Vvar = np.sum(data * data, axis=0)
        rmat = np.tile(np.arange(0, Nu), (Nt, 1)).T

        w = np.zeros((Nu, Nt))
        w[(rmat == labels)] = 1
        e = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2) / (Nt * (Ne - 1))
        window = np.ones((1, 2 * half_window_size + 1))

        S0 = 0
        while True:
            Nb = convolve2d(w, window, mode="same")
            x = (np.tile(Vvar, (Nu, 1)) - (np.dot(states, data)) ** 2) / (
                2 * e * (Ne - 1)
            ) - factor * Nb
            dlt = np.argmin(x, axis=0)

            labels = dlt
            w = np.zeros((Nu, Nt))
            w[(rmat == labels)] = 1
            Su = np.sum(Vvar - np.sum(np.dot(w.T, states).T * data, axis=0) ** 2) / (Nt * (Ne - 1))
            if np.abs(Su - S0) <= np.abs(tol * Su):
                break
            S0 = Su

        return labels

    @staticmethod
    def _median_smooth(labels, n_win, nbr_clust):
        """
        Median filter
        :param labels:
        :param n_win:
        :param nbr_clust:
        :return:
        """
        labels_copy = None
        nbr_microstates = None
        for w in n_win:
            labels_copy = np.copy(labels)
            for l in range(w, len(labels) - w):
                s = np.array([np.sum(labels[l - w : l + w] == j) for j in range(nbr_clust)])
                labels_copy[l] = np.argmax(s)
            nbr_microstates = get_nbr_microstates(labels_copy)
            if nbr_microstates == nbr_clust:
                break
        labels = labels_copy
        if len(n_win) > 1 and nbr_microstates != nbr_clust:
            raise ValueError(
                f"The 'strict' smoothing could not produce '{nbr_clust}' distinct clusters"
            )
        return labels

    @staticmethod
    def _reject_short_segments(
        segmentation: NDArray[int],
        data: NDArray[float],
        min_segment_length: int,
    ) -> NDArray[int]:
        """Reject segments that are too short.

        Reject segments that are too short by replacing the labels with the
        adjacent labels based on data correlation.
        """
        while True:
            # list all segments (groupby = returns consecutive keys and groups from the iterable)
            segments = [list(group) for _, group in groupby(segmentation)]
            idx = 0  # where does the segment start

            for k, segment in enumerate(segments):
                skip_condition = [
                    k in (0, len(segments) - 1),  # ignore edge segments
                    segment[0] == -1,  # ignore segments labelled with 0
                    min_segment_length <= len(segment),  # ignore large segments
                ]
                if any(skip_condition):
                    idx += len(segment)
                    continue

                left = idx
                right = idx + len(segment) - 1
                new_segment = segmentation[left : right + 1]

                while len(new_segment) != 0:
                    # compute correlation left/right side
                    left_corr = np.abs(
                        compute_corr(
                            data[:, :, left - 1].T,
                            data[:, :, left].T,
                        )
                    )
                    right_corr = np.abs(compute_corr(data[:, :, right].T, data[:, :, right + 1].T))

                    if np.abs(right_corr - left_corr) <= 1e-8:
                        # equal corr, try to do both sides
                        if len(new_segment) == 1:
                            # do only one side, left
                            segmentation[left] = segmentation[left - 1]
                            left += 1
                        else:
                            # If equal, do both sides
                            segmentation[right] = segmentation[right + 1]
                            segmentation[left] = segmentation[left - 1]
                            right -= 1
                            left += 1
                    else:
                        if left_corr < right_corr:
                            segmentation[right] = segmentation[right + 1]
                            right -= 1
                        elif right_corr < left_corr:
                            segmentation[left] = segmentation[left - 1]
                            left += 1

                    # crop segment
                    new_segment = segmentation[left : right + 1]

                # segments that were too short might have become long enough,
                # so list them again and check again.
                break
            else:
                break  # stop while loop because all segments are long enough

        return segmentation

    @staticmethod
    def _reject_edge_segments(segmentation: NDArray[int]) -> NDArray[int]:
        """Set the first and last segment as unlabeled (0).  => make the 1st and last microstate = -1"""
        # set first segment to unlabeled
        n = (segmentation != segmentation[0]).argmax()
        segmentation[:n] = -1

        # set last segment to unlabeled
        n = np.flip((segmentation != segmentation[-1])).argmax()
        segmentation[-n:] = -1

        return segmentation

    def save_centroids(self, model_centroids_path):
        with open(model_centroids_path, "wb") as f1:
            pickle.dump(self._centroids, f1)
            print(f"file {model_centroids_path} saved")

    # --------------------------------------------------------------------
    @staticmethod
    def _check_n_clusters(n_clusters: int) -> int:
        """Check that the number of clusters is a positive integer."""
        _check_type(n_clusters, ("int",), item_name="n_clusters")
        if n_clusters <= 0:
            raise ValueError(
                "The number of clusters must be a positive integer. " f"Provided: '{n_clusters}'."
            )
        return n_clusters

    @staticmethod
    def _check_max_iter(max_iter: int) -> int:
        """Check that max_iter is a positive integer."""
        _check_type(max_iter, ("int",), item_name="max_iter")
        if max_iter <= 0:
            raise ValueError(
                "The number of max iteration must be a positive integer. "
                f"Provided: '{max_iter}'."
            )
        return max_iter

    @staticmethod
    def _check_n_runs(n_runs: int) -> int:
        """Check that n_runs is a positive integer."""
        _check_type(n_runs, ("int",), item_name="n_runs")
        if n_runs <= 0:
            raise ValueError(f"The tolerance must be a positive integer. Provided: '{n_runs}'.")
        return n_runs

    @staticmethod
    def _check_tol(tol: Union[int, float]) -> Union[int, float]:
        """Check that tol is a positive number."""
        _check_type(tol, ("numeric",), item_name="tol")
        if tol <= 0:
            raise ValueError(f"The tolerance must be a positive number. Provided: '{tol}'.")
        return tol

    # --------------------------------------------------------------------

    @property
    def max_iter(self) -> int:
        """Maximum number of iterations of the k-means algorithm for a run.

        :type: `int`
        """
        return self._max_iter

    @property
    def n_runs(self) -> int:
        """Number of times of running the algorithm with different centroid's initialization.

        :type: `int`
        """
        return self._n_runs

    @property
    def tol(self) -> int:
        """The within-cluster variation metric used to declare convergence

        :type: `int`
        """
        return self._tol

    @property
    def _gev(self) -> np.ndarray:
        """Global Explained Variance.

        :type: `ndarray`
        """
        if self._gev_ is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._gev_

    @property
    def gev_max(self) -> np.ndarray:
        """Global Explained Variance max.

        :type: `ndarray`
        """
        if self._gev_max is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._gev_max

    @property
    def n_clust(self) -> int:
        """Number of clusters (number of microstates).

        :type: `int`
        """
        return self._n_clust

    @property
    def best_run(self) -> int:
        """Run corresponding to highest GEV

        :type: `int`
        """
        return self._best_run

    @property
    def centroids(self) -> NDArray[float]:
        """Fitted clusters (the microstates connectivity graphs).

        Returns None if cluster algorithm has not been fitted.

        :type: `~numpy.array` of shape (n_clusters, n_channels, n_channels) | None
        """
        if self._centroids is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._centroids.copy()

    def _all_centroids_by_run(self) -> NDArray[float]:
        """Fitted clusters (the microstates connectivity graphs) by runs.

        Returns None if cluster algorithm has not been fitted.

        :type: `~numpy.array` of shape (n_runs, n_clusters, n_channels, n_channels) | None
        """
        if self._all_centroids_by_run_ is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._all_centroids_by_run_.copy()

    @property
    def fitted_data(self) -> NDArray[float]:
        """Data array used to fit the clustering algorithm.

        :type: `~numpy.array` of shape (n_channels, n_channels, n_samples) | None
        """
        if self._fitted_data is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._fitted_data.copy()

    @property
    def _run_labels(self) -> NDArray[int]:
        """Microstate label attributed to each sample of the fitted data.

        :type: `~numpy.array` of shape (n_samples, ) | None
        """
        if self._run_labels_ is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._run_labels_.copy()

    @property
    def cluster_names(self) -> List[str]:
        """Name of the clusters.

        :type: `list`
        """
        return self._cluster_names.copy()

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

    @property
    def converg_iter(self) -> NDArray[int]:
        """Array containing the number of iterations needed to fit (convergence) for each run

        :type: `~numpy.array` of shape (n_runs, ) | None
        """
        if self._converg_iter is None:
            assert not self._fitted  # sanity-check
            print("Clustering algorithm has not been fitted.")
        return self._converg_iter.copy()

    @property
    def fitted(self) -> bool:
        """Fitted state.

        :type: `bool`
        """
        return self._fitted

    @fitted.setter
    def fitted(self, fitted):
        """Property-setter used to reset all fit variables."""
        _check_type(fitted, (bool,), item_name="fitted")
        if fitted and not self._fitted:
            print(
                "The property 'fitted' can not be set to 'True' directly. "
                "Please use the .fit() method to fit the clustering algorithm."
            )
        elif fitted and self._fitted:
            print(
                "The property 'fitted' can not be set to 'True' directly. "
                "The clustering algorithm has already been fitted."
            )
        else:
            self._centroids = None
            self._fitted_data = None
            self._run_labels_ = None
            self._fitted = False
