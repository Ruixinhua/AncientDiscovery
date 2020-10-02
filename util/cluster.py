from util import tools
import pandas as pd
import numpy as np
from sklearn.cluster import estimate_bandwidth, MeanShift


def run_cluster(label, char_type, group_df, quantile=0.8, add_center=True):
    group_df = group_df.reset_index(drop=True)
    # group images data by label
    feature = np.array(group_df.loc[:, "feature"])
    feature = np.stack(feature)
    # bandwidth, it is the search radius of when take specific point as core
    bandwidth = estimate_bandwidth(feature, quantile=quantile, n_samples=feature.shape[0])
    bandwidth = 0.0001 if bandwidth <= 0.0001 else bandwidth
    # set mean shift function here
    ms = MeanShift(bandwidth=bandwidth, max_iter=1000, n_jobs=-1)
    # training
    ms.fit(feature)
    cluster_centers = np.array(ms.cluster_centers_)
    group_df["center"] = list(ms.labels_)
    group_df["size"] = 1
    if add_center:
        centers_df = pd.DataFrame({"feature": list(cluster_centers), "center": list(range(len(cluster_centers)))})
        centers_df["size"] = centers_df.center.apply(lambda p: sum(np.array(ms.labels_) == p))
        centers_df["label"], centers_df["type"] = label, char_type
        group_df = group_df.append(centers_df, ignore_index=True, sort=False)
    return group_df


def get_cluster_output_df(input_df=None, add_center=True):
    """
    get cluster result and store the result
    Args:
        input_df: A pandas data frame with columns “label", "type", "feature"
        add_center: Whether add center point to final results

    Returns: A pandas data frame with columns “label", "type", "feature", "center", "size"

    """
    if input_df is None:
        tools.print_log("No input")
        return
    columns = input_df.columns
    output_df = pd.DataFrame(columns=columns)
    for (label, char_type), group_df in input_df.groupby(["label", "type"]):
        output_df = output_df.append(run_cluster(label, char_type, group_df, add_center=add_center), ignore_index=True,
                                     sort=False)
    tools.print_log("shape of output after cluster: %s" % str(output_df.shape))
    return output_df

