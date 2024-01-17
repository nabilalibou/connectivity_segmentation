import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from matplotlib.axes import Axes
from connectivity_segmentation.utils.checks import _check_type


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def plot_segmentation(
    labels,
    gcp,
    times,
    list_clusters,
    cluster_names=None,
    axes=None,
    cbar_axes=None,
    show=True,
    block=False,
):
    _check_type(labels, (np.ndarray,), "labels")  # 1D array (n_times, )
    _check_type(gcp, (np.ndarray,), "gcp")  # 1D array (n_times, )
    _check_type(times, (np.ndarray,), "times")  # 1D array (n_times, )
    _check_type(cluster_names, (None, list, tuple), "cluster_names")
    _check_type(axes, (None, Axes), "ax")
    _check_type(cbar_axes, (None, Axes), "cbar_ax")
    if cluster_names:
        clustNameProvided = True
    else:
        clustNameProvided = False
    if len(list_clusters) <= 0 or not all(val > 0 for val in list_clusters):
        raise ValueError(
            f"Provided number of clusters {len(list_clusters)} is invalid. The number "
            "of clusters must be strictly positive."
        )

    if labels.shape[0] != len(list_clusters):
        raise ValueError(
            f"Dimension of Labels array need to be: n_clusters {len(list_clusters)} x times {times.shape[0]}"
        )

    if axes is None:
        if len(list_clusters) > 1:
            for col in range(5, 0, -1):
                for row in range(3, 0, -1):
                    if len(list_clusters) % (row * col) < len(list_clusters):
                        fig, axes = plt.subplots(
                            row,
                            col,
                            figsize=(col * (16 / 5) + 4, row * (8 / 3) + 2),
                            sharex="col",
                            sharey="row",
                            # gridspec_kw={'height_ratios': [1, 1]}
                        )
                        break
                else:
                    continue
                break
        else:
            fig, axes = plt.subplots(1, 1)
    else:
        fig = axes.get_figure()

        # if not isinstance(axes, np.ndarray):
        #     axes = np.array(axes)
    # if not isinstance(axes, list):
    #     axes = [axes]

    for k, axe in enumerate(axes):
        L = labels[k]

        # check cluster_names
        if not clustNameProvided:
            cluster_names = [str(n) for n in range(1, list_clusters[k] + 1)]
        if len(cluster_names) != list_clusters[k]:
            raise ValueError(
                "Argument 'cluster_names' should have the 'n_clusters' elements. "
                f"Provided: {len(cluster_names)} names for {list_clusters[k]} clusters."
            )

        # define states and colors
        state_labels = [-1] + list(range(list_clusters[k]))
        cluster_names = ["∅"] + cluster_names
        n_colors = list_clusters[k] + 1
        # plot
        axe.plot(times, gcp, color="black", linewidth=0.2)
        # cmap = plt.cm.get_cmap(colormaps["viridis"], n_colors) does not work for matplotlib 3.6
        cmap = colormaps["viridis"].resampled(n_colors)
        for state, color in zip(state_labels, cmap.colors):
            pos = np.where(L == state)[0]
            if len(pos):
                pos = np.unique([pos, pos + 1])
                x = np.zeros(L.shape).astype(bool)
                if pos[-1] == L.size:
                    pos = pos[:-1]
                x[pos] = True
                axe.fill_between(times, gcp, color=color, where=x, step=None, interpolate=False)

        # format
        axe.set_xlabel("Time (s)")
        axe.set_ylabel("Global Connectivity Power (PLV)")
        axe.set_title(f"{list_clusters[k]} clusters")
        if k >= len(list_clusters) - 1:
            # color bar
            norm = colors.Normalize(vmin=0, vmax=n_colors)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            colorbar = fig.colorbar(
                sm, cax=cbar_axes, ax=axe, ticks=[i + 0.5 for i in range(n_colors)]
            )
            colorbar.ax.set_yticklabels(cluster_names)
            break
        if not clustNameProvided:
            cluster_names = None

    # common formatting
    fig.suptitle("Segmentation")
    plt.autoscale(tight=True)

    if show:
        plt.show(block=block)
    return fig
