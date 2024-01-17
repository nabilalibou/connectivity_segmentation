import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")


def save_param_csv(
    params_values, params_name, K_range, params_path, first_col="Statistic name / Clust"
):
    it = iter(params_values)
    the_len = len(next(it))  # len of the next item from the iterator
    if not all(len(l) == the_len for l in it):
        raise ValueError(f"not all lists in 'params_values' have same length!")
    if the_len != len(params_name):
        raise ValueError(
            f"Number of parameter values in the list of 'params_values' and 'params_name' are not equal: "
            f"'{params_values}' vs '{params_name}'"
        )
    else:
        params_nbr = len(params_values)
    csv_fieldnames = [f"K = {k}" for k in K_range]
    csv_fieldnames.insert(0, first_col)
    stats_dict = {}
    for k in range(params_nbr):
        for param in range(len(params_name)):
            try:
                stats_dict[first_col].append(params_name[param])
                stats_dict[f"{csv_fieldnames[int(k) + 1]}"].append(f"{params_values[k][param]}")
            except KeyError:
                stats_dict[first_col] = [params_name[param]]
                stats_dict[f"{csv_fieldnames[int(k) + 1]}"] = [f"{params_values[k][param]}"]
    stats_df = pd.DataFrame(stats_dict)
    writer = pd.ExcelWriter(params_path)
    stats_df.to_excel(writer, sheet_name="my_analysis", index=False, header=True, na_rep="NaN")
    # Auto-adjust columns' width
    for column in stats_df:
        column_width = max(stats_df[column].astype(str).map(len).max(), len(column))
        col_idx = stats_df.columns.get_loc(column)
        writer.sheets["my_analysis"].set_column(col_idx, col_idx, column_width)
    writer.close()
    print(f"file {params_path} saved")
    return stats_dict


def save_group_stats(
    stats_values, stats_name, K_range, stats_path, first_col="Statistic name / Clust"
):
    """
    :param stats_values:
    :param stats_name:
    :param stats_path:
    :return:
    """
    csv_fieldnames = [f"K = {k}" for k in K_range]
    csv_fieldnames.insert(0, first_col)
    stats_dict = {}
    for v, stat in enumerate(range(stats_values.shape[2])):
        try:
            stats_dict[first_col].append(stats_name[v])
        except KeyError:
            stats_dict[first_col] = [stats_name[v]]
        for k in range(stats_values.shape[1]):
            if np.isnan(stats_values[0, k, stat]):
                try:
                    stats_dict[f"{csv_fieldnames[k + 1]}"].append("NaN")
                except KeyError:
                    stats_dict[f"{csv_fieldnames[k + 1]}"] = ["NaN"]
            else:
                val_mean = np.mean(stats_values[:, k, stat], axis=0).round(3)
                val_std = np.std(stats_values[:, k, stat], axis=0).round(3)
                try:
                    stats_dict[f"{csv_fieldnames[k + 1]}"].append(f"{val_mean}±{val_std}")
                except KeyError:
                    stats_dict[f"{csv_fieldnames[k + 1]}"] = [f"{val_mean}±{val_std}"]
    stats_df = pd.DataFrame(stats_dict)
    writer = pd.ExcelWriter(stats_path)
    stats_df.to_excel(writer, sheet_name="my_analysis", index=False, header=True, na_rep="NaN")
    # Auto-adjust columns' width
    for column in stats_df:
        column_width = max(stats_df[column].astype(str).map(len).max(), len(column))
        col_idx = stats_df.columns.get_loc(column)
        writer.sheets["my_analysis"].set_column(col_idx, col_idx, column_width)
    writer.close()
    print(f"file {stats_path} saved")
    return stats_dict


def save_outliers_catalog(
    stats_values,
    file_names,
    group_stats_dict,
    outliers_catalog_path,
    first_col="Statistic name / Clust",
):
    """
    :param stats_values:
    :param file_names:
    :param group_stats_dict:
    :param outliers_catalog_path:
    :return:
    """
    outliers_dict = {"file_names": {}, "categories": {}}
    for f in range(stats_values.shape[0]):
        for k in range(stats_values.shape[1]):
            file_occurence = 0
            for stat_idx in range(stats_values.shape[2]):
                clustcnt = 0
                for row_name, val_list in group_stats_dict.items():
                    if row_name != first_col:
                        if clustcnt == k:
                            clustcnt += 1
                            try:
                                if (
                                    outliers_dict["categories"][row_name]
                                    and outliers_dict["file_names"][row_name]
                                ):
                                    pass
                            except KeyError:
                                outliers_dict["categories"][row_name] = {}
                                outliers_dict["file_names"][row_name] = {}
                            if val_list[stat_idx] != "NaN":
                                mean, std = val_list[stat_idx].split("±")
                                if abs(stats_values[f, k, stat_idx]) > float(mean) + 2 * float(std):
                                    stat_name = group_stats_dict[first_col][stat_idx]
                                    try:
                                        outliers_dict["categories"][row_name][stat_name][
                                            file_names[f]
                                        ] = stats_values[f, k, stat_idx]
                                        outliers_dict["file_names"][row_name][
                                            f"{file_names[f]}"
                                        ] += 1
                                    except KeyError:
                                        outliers_dict["categories"][row_name][stat_name] = {
                                            file_names[f]: stats_values[f, k, stat_idx]
                                        }
                                        outliers_dict["file_names"][row_name][
                                            f"{file_names[f]}"
                                        ] = file_occurence
                        else:
                            clustcnt += 1
                            pass
    with open(outliers_catalog_path, "w") as outfile:
        json.dump(outliers_dict, outfile, indent=4)
    print(f"file {outliers_catalog_path} saved")


def save_group_stats(
    stats_values, stats_name, K_range, stats_path, first_col="Statistic name / Clust"
):
    """
    :param stats_values:
    :param stats_name:
    :param stats_path:
    :return:
    """
    csv_fieldnames = [f"K = {k}" for k in K_range]
    csv_fieldnames.insert(0, first_col)
    stats_dict = {}
    for v, stat in enumerate(range(stats_values.shape[2])):
        try:
            stats_dict[first_col].append(stats_name[v])
        except KeyError:
            stats_dict[first_col] = [stats_name[v]]
        for k in range(stats_values.shape[1]):
            if np.isnan(stats_values[0, k, stat]):
                try:
                    stats_dict[f"{csv_fieldnames[k + 1]}"].append("NaN")
                except KeyError:
                    stats_dict[f"{csv_fieldnames[k + 1]}"] = ["NaN"]
            else:
                val_mean = np.mean(stats_values[:, k, stat], axis=0).round(3)
                val_std = np.std(stats_values[:, k, stat], axis=0).round(3)
                try:
                    stats_dict[f"{csv_fieldnames[k + 1]}"].append(f"{val_mean}±{val_std}")
                except KeyError:
                    stats_dict[f"{csv_fieldnames[k + 1]}"] = [f"{val_mean}±{val_std}"]
    stats_df = pd.DataFrame(stats_dict)
    writer = pd.ExcelWriter(stats_path)
    stats_df.to_excel(writer, sheet_name="my_analysis", index=False, header=True, na_rep="NaN")
    # Auto-adjust columns' width
    for column in stats_df:
        column_width = max(stats_df[column].astype(str).map(len).max(), len(column))
        col_idx = stats_df.columns.get_loc(column)
        writer.sheets["my_analysis"].set_column(col_idx, col_idx, column_width)
    writer.close()
    print(f"file {stats_path} saved")
    return stats_dict


# def save_avg_segm(labels_array):
# All labels need to only have K segments or else error => need more smoothing. See code to check how many segments
# they contain
# if show_std = True, compute start, stop + std and add vertical dashed line.
#     for f in range(labels_array.shape[0]):
#         label_duration_f = []
#         for k in range(labels_array.shape[1]):
#             # getting Consecutive elements
#             res = []
#             cnt = 1
#             label_duration = []
#             for idx in range(labels.shape[2]):
#                 if labels[idx] == k and labels[idx] == labels[idx + 1]:
#                     cnt += 1
#                 elif labels[idx] == k:
#                     res.append(cnt)
#                 if idx == labels.shape[0] - 2 and labels[idx] == k:
#                     res.append(cnt)
#                     break
#             label_duration.append(res)
#         label_duration_f.append(label_duration)
#
#     labels_duration_f = np.array(label_duration_f)
