import numpy as np
import pandas as pd

def combine_bbox(bboxes: list[list[float]]):
    numpy_bbox_matrix = np.array(bboxes)
    return [
        numpy_bbox_matrix[:, 0].min(),
        numpy_bbox_matrix[:, 1].min(),
        numpy_bbox_matrix[:, 0].max(),
        numpy_bbox_matrix[:, 1].max(),
    ]


def get_bins_from_list(cord_list: list[float], bins:int=100):
    return pd.cut(cord_list, bins=bins, labels=np.arange(1, bins+1)).to_list()