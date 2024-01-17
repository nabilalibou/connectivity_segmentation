import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha="center")


def plot_cluster_scores(scores_dict, K_range, show=True):
    # Evaluating clustering fits and compare scores. If runs > 1 plot consistency as well
    x_scores = []
    y_scores = []
    for key, val in scores_dict.items():
        x_scores.append(key)
        val = [round(n, 2) for n in val]
        y_scores.append(val)

    # invert davies-bouldin scores
    scores_dict["Davies-Bouldin"] = 1 / (1 + scores_dict["Davies-Bouldin"])

    # normalize scores using sklearn
    from sklearn.preprocessing import normalize

    scores_dict = {
        score: normalize(value[:, np.newaxis], axis=0).ravel()
        for score, value in scores_dict.items()
    }

    # set width of a bar and define colors
    barWidth = 0.2
    colors = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F"]

    # create figure
    plt.figure(figsize=(10, 8))
    # create the position of the bars on the X-axis
    x = [[elt + k * barWidth for elt in np.arange(len(K_range))] for k in range(len(scores_dict))]
    # create plots
    for k, (score, values) in enumerate(scores_dict.items()):
        rects = plt.bar(
            x=x[k],
            height=values,
            width=barWidth,
            edgecolor="grey",
            color=colors[k],
            label=score,
        )
        # addlabels(K_range, y_scores[k])
        plt.bar_label(rects, labels=y_scores[k])  # padding=3
    # add labels and legend
    plt.xlabel("Number of clusters")
    plt.ylabel("Score normalized to unit norm")
    plt.xticks(
        [pos + 1.5 * barWidth for pos in range(len(K_range))],
        [str(k) for k in K_range],
    )
    plt.legend()
    plt.title("Clustering scores per number of clusters (with real score values displayed)")
    if show:
        plt.show()
    return plt
