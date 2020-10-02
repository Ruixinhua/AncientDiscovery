
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

from util import tools


def get_reduction_result(input_df=None):
    """
        get reduction result from input DataFrame object

    Args:
        input_df: A pandas data frame with columns “label", "type", "feature"

    Returns:  pandas data frame with columns “label", "type", "feature" and "feature" column is 2-D.

    """
    if input_df is None:
        tools.print_log("No input")
        return
    columns = input_df.columns
    output_df = pd.DataFrame(columns=columns)
    # take the feature columns
    feature = input_df["feature"].values
    feature = np.stack(feature)
    fea_dim = feature[0].shape[0]
    if fea_dim > 512:
        # use PCA to reduce the dimensions
        pca = PCA(n_components=256)
        feature = pca.fit_transform(feature)
        tools.print_log("Variance of pca: %s" % str(np.sum(pca.explained_variance_ratio_)))
    # reduce dimension to 2-D
    feature_reduction = TSNE(n_components=2, n_iter=1000, random_state=42).fit_transform(feature)
    input_df = input_df.drop(columns=["feature"])
    input_df["feature"] = list(feature_reduction)
    feature_reduction = np.array(feature_reduction)
    tools.print_log("Shape of features after reduce dimension: %s" % str(feature_reduction.shape))
    output_df = output_df.append(input_df, ignore_index=True)
    return output_df
