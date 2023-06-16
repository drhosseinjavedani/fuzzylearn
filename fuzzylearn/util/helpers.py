import itertools
import math
from statistics import mean, mode

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def fuzzify_info(*args, **kwargs):
    """An internal function to get a  fuzzifying info as a dictinary"""

    split_dict = {}
    data = kwargs["data"]
    if isinstance(data, pd.DataFrame):
        data, columns = pd_to_np(data=data)
        data = data.T
    elif isinstance(data, np.ndarray):
        data = data.T
    else:
        raise ValueError(f"{type(data)} is not supported!")

    number_of_intervals = kwargs["number_of_intervals"]
    index = 0
    n = len(data)
    for col in data:
        min_col = col.min()
        max_col = col.max()
        if np.all(np.mod(col, 1) == 0) and len(np.unique(col)) <= 5:
            ordered_list = sorted(col.tolist())
            subtracted_list = [
                ordered_list[i] - ordered_list[i - 1]
                for i in range(1, len(ordered_list))
            ]
            len_col = min(subtracted_list) / 2.0
        else:
            if isinstance(number_of_intervals, int):
                len_col = (max_col - min_col) / number_of_intervals
            if isinstance(number_of_intervals, list):
                if all([isinstance(item, int) for item in number_of_intervals]):
                    len_col = (max_col - min_col) / number_of_intervals[index]
            if number_of_intervals == "sturges":
                len_col = (math.log(n, 2)) + 1
            if number_of_intervals == "rice":
                len_col = 2 * n**1 / 3
            if number_of_intervals == "freedman":
                q3, q1 = np.percentile(data.T[:, index], [75, 25])
                iqr = q3 - q1
                h = 2 * iqr / (n**1 / 3)
                len_col = (max_col - min_col) / h
        split_dict[index] = [min_col, max_col, len_col]
        index += 1
    return split_dict


def np_to_pd(*args, **kwargs):
    """An internal function to transform pandas dataframe to 2D numpy array"""

    columns = kwargs["columns"]
    arr = kwargs["arr"]
    features = columns.to_list()
    df = pd.DataFrame(arr, columns=columns)
    return df


def pd_to_np(*args, **kwargs):
    """An internal function to transform 2D numpy array to pandas dataframe"""

    df = kwargs["data"]
    columns = df.columns
    features = columns.to_list()
    return df.to_numpy(), columns


def process_train_data(*args, **kwargs):
    """An internal function to return X_train and y_train"""

    X_train = kwargs["X_train"]
    y_train = kwargs["y_train"]
    X_valid = kwargs["X_valid"]
    y_valid = kwargs["y_valid"]

    if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
        if isinstance(X_valid, np.ndarray) and isinstance(y_valid, np.ndarray):
            return X_train, y_train, X_valid, y_valid
        elif X_valid is None or y_valid is None:
            return X_train, y_train, X_valid, y_valid
        else:
            raise ValueError("X_valid or y_valid are not in proper types !")
    if X_train is not None and isinstance(X_train, pd.DataFrame):
        X_train = X_train.reset_index(drop=True)
    if y_train is not None and isinstance(y_train, pd.DataFrame):
        y_train = y_train.reset_index(drop=True)
        y_for_color_code = y_train
    if X_valid is not None and isinstance(X_valid, pd.DataFrame):
        X_valid = X_valid.reset_index(drop=True)
    if y_valid is not None and isinstance(y_valid, pd.DataFrame):
        y_valid = y_valid.reset_index(drop=True)

    return X_train, y_train, X_valid, y_valid


# TODO check all
def fuzzifying(*args, **kwargs):
    """An internal function to get a pandas dataframe and return fuzzifying of its variables"""

    fuzzy_type = kwargs["fuzzy_type"]
    fuzzy_cut = kwargs["fuzzy_cut"]
    X = kwargs["X"]
    if isinstance(X, pd.DataFrame):
        X, columns = pd_to_np(data=X)
    elif isinstance(X, np.ndarray):
        X = X
    else:
        raise ValueError(f"{type(X)} is not supported!")

    zero_array = np.zeros_like(X)
    XX = np.hstack((zero_array, zero_array))
    X_copy = np.copy(X)
    if fuzzy_type == "simple":
        number_of_intervals = kwargs["number_of_intervals"]
        split_dict = fuzzify_info(data=X, number_of_intervals=number_of_intervals)
        # row_i iterate in rows of X_transpose
        for row_i in range(X.T.shape[0]):
            if split_dict[row_i][2] != 0:
                X_copy[:, row_i] = (
                    ((X[:, row_i] - split_dict[row_i][0]) / split_dict[row_i][2])
                    .round(decimals=0)
                    .astype(int)
                )
        return X_copy
    # TODO has problem
    if fuzzy_type == "deep":
        number_of_intervals = kwargs["number_of_intervals"]
        split_dict = fuzzify_info(data=X, number_of_intervals=number_of_intervals)
        # row_i iterate in rows of X_transpose
        col_j = 0
        for row_i in range(X.shape[0]):
            for col_j in range(X.shape[1]):
                if split_dict[col_j][2] != 0:
                    if (
                        np.modf(
                            (X[row_i, col_j] - split_dict[col_j][0])
                            / split_dict[col_j][2]
                        )[0]
                        >= 0.5
                    ):
                        XX[row_i, 2 * col_j] = (
                            (
                                (X[row_i, col_j] - split_dict[col_j][0])
                                / split_dict[col_j][2]
                            )
                            .round(decimals=0)
                            .astype(int)
                        )
                        XX[row_i, 2 * col_j + 1] = (
                            (X[row_i, col_j] - split_dict[col_j][0])
                            / split_dict[col_j][2]
                        ) + split_dict[col_j][2] / 2
    if fuzzy_type == "triangular":
        X_copy = np.copy(X)
        number_of_intervals = kwargs["number_of_intervals"]
        split_dict = fuzzify_info(data=X_copy, number_of_intervals=number_of_intervals)
        # row_i iterate in rows of X_transpose
        col_j = 0
        for row_i in range(X.shape[0]):
            for col_j in range(X.shape[1]):
                if split_dict[col_j][2] != 0:
                    a = (
                        (
                            (X[row_i, col_j] - split_dict[col_j][0])
                            / split_dict[col_j][2]
                        )
                        .round(decimals=0)
                        .astype(int)
                    )
                    b = a + 1
                    membership_coef = (
                        (
                            (X[row_i, col_j] - split_dict[col_j][0])
                            / split_dict[col_j][2]
                        )
                        - a
                    ) / (b - a)
                    if (
                        membership_coef >= fuzzy_cut
                        and membership_coef <= 1 - fuzzy_cut
                    ):
                        X_copy[row_i, col_j] = a
                    if membership_coef < fuzzy_cut:
                        X_copy[row_i, col_j] = (a + (a - 1)) / 2
                    if membership_coef > 1 - fuzzy_cut:
                        X_copy[row_i, col_j] = (a + (a + 1)) / 2

        return X_copy


def lhs_rhs_creator(*args, **kwargs):
    trained = {}
    X = kwargs["X"]
    y = kwargs["y"]
    X_train_F = kwargs["X_train_F"]
    smaller_better = kwargs["smaller_better"]
    metric = kwargs["metric"]
    threshold = kwargs["threshold"]

    X_paired_weights = pairwise_distances(X, X, metric=metric)
    if smaller_better:
        X_paired_weights[X_paired_weights > threshold] = np.inf
    else:
        X_paired_weights[X_paired_weights < threshold] = np.inf

    i_index = 0
    lhss = np.empty(shape=X.shape)
    rhss = np.empty(shape=y.shape)
    for row in X_paired_weights:
        lhs = []
        rhs = []
        j_index = 0
        for member in row:
            if member != np.inf:
                lhs.append(X_train_F[j_index, :])
                rhs.append(y[j_index])
            j_index += 1
        temp = np.mean(lhs, 0).tolist()
        lhss[i_index] = temp
        try:
            rhs = list(itertools.chain(*rhs))
            rhss[i_index] = mode(rhs)
        except Exception as e:
            # TODO check this line for np arrays
            rhss[i_index] = mode(rhs)
        i_index += 1

    return lhss, rhss


def lhs_rhs_creator_regression(*args, **kwargs):
    trained = {}
    X = kwargs["X"]
    y = kwargs["y"]
    X_train_F = kwargs["X_train_F"]
    smaller_better = kwargs["smaller_better"]
    metric = kwargs["metric"]
    threshold = kwargs["threshold"]

    X_paired_weights = pairwise_distances(X, X, metric=metric)
    if smaller_better:
        X_paired_weights[X_paired_weights > threshold] = np.inf
    else:
        X_paired_weights[X_paired_weights < threshold] = np.inf

    i_index = 0
    lhss = np.empty(shape=X.shape)
    rhss = np.empty(shape=y.shape)
    for row in X_paired_weights:
        lhs = []
        rhs = []
        j_index = 0
        for member in row:
            if member != np.inf:
                lhs.append(X_train_F[j_index, :])
                rhs.append(y[j_index])
            j_index += 1
        temp = np.mean(lhs, 0).tolist()
        lhss[i_index] = temp
        try:
            rhs = list(itertools.chain(*rhs))
            rhss[i_index] = mean(rhs)
        except Exception as e:
            # TODO check this line for np arrays
            rhss[i_index] = mean(rhs)
        i_index += 1

    return lhss, rhss


def trained_model_for_X_y(*args, **kwargs):
    """training function."""
    X = kwargs["X"]
    y = kwargs["y"]
    X_train_F = kwargs["X_train_F"]
    metric = kwargs["metric"]
    threshold = kwargs["threshold"]
    smaller_better = kwargs["smaller_better"]

    if isinstance(X, pd.DataFrame):
        X, X_cols = pd_to_np(data=X)
    if isinstance(y, pd.DataFrame):
        y, y_cols = pd_to_np(data=y)

    lhss, rhss = lhs_rhs_creator(
        smaller_better=smaller_better,
        X_train_F=X_train_F,
        X=X,
        y=y,
        metric=metric,
        threshold=threshold,
    )

    return lhss, rhss
