from statistics import mode

import numpy as np
from sklearn.metrics import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

from fuzzylearn.util.helpers import *
from fuzzylearn.util.read_data import read_yaml_file


class FLRegressor:
    """FuzzyLearning class"""

    def __init__(self, *args, **kwargs):
        self.number_of_intervals = kwargs["number_of_intervals"]
        self.metric = kwargs["metric"]
        self.threshold = kwargs["threshold"]
        self.fuzzy_type = kwargs["fuzzy_type"]
        self.trained = {}
        self.X_paired_weights = self.output_weights = None
        self.X_train_F = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.smaller_better = True
        self.rhss = None
        self.lhss = None
        self.features = None
        self.y_for_color_code = None
        self.metric_for_optimum = None
        self.threshold_for_optimum = None
        self.number_of_intervals_for_optimum = None
        try:
            self.n_trials = kwargs["n_trials"]
        except Exception as e:
            self.n_trials = None
        try:
            self.fuzzy_cut = kwargs["fuzzy_cut"]
        except Exception as e:
            self.fuzzy_cut = None

    def __str__(self):
        return f"number_of_intervals :{self.number_of_intervals} \n metric : {self.metric} \n threshold: {self.threshold} \n "

    @property
    def number_of_intervals_for_optimum(self):
        return self._number_of_intervals_for_optimum

    @number_of_intervals_for_optimum.setter
    def number_of_intervals_for_optimum(self, value):
        self._number_of_intervals_for_optimum = value

    @property
    def threshold_for_optimum(self):
        return self._threshold_for_optimum

    @threshold_for_optimum.setter
    def threshold_for_optimum(self, value):
        self._threshold_for_optimum = value

    @property
    def metric_for_optimum(self):
        return self._metric_for_optimum

    @metric_for_optimum.setter
    def metric_for_optimum(self, value):
        self._metric_for_optimum = value

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def y_for_color_code(self):
        return self._y_for_color_code

    @y_for_color_code.setter
    def y_for_color_code(self, value):
        self._y_for_color_code = value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def X_valid(self):
        return self._X_valid

    @X_valid.setter
    def X_valid(self, value):
        self._X_valid = value

    @property
    def y_valid(self):
        return self._y_valid

    @y_valid.setter
    def y_valid(self, value):
        self._y_valid = value

    @property
    def metric_for_optimization(self):
        return self._metric_for_optimization

    @metric_for_optimization.setter
    def metric_for_optimization(self, value):
        self._metric_for_optimization = value

    @property
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        self._n_trials = value

    @property
    def fuzzy_type(self):
        return self._fuzzy_type

    @fuzzy_type.setter
    def fuzzy_type(self, value):
        self._fuzzy_type = value

    @property
    def fuzzy_cut(self):
        return self._fuzzy_cut

    @fuzzy_cut.setter
    def fuzzy_cut(self, value):
        self._fuzzy_cut = value

    def fit(self, *args, **kwargs):
        """Fit function."""

        # arguments for the fit
        self.X_valid = kwargs["X_valid"]
        self.y_valid = kwargs["y_valid"]
        self.X_train = kwargs["X"]
        self.y_train = kwargs["y"]

        X_train, y_train, X_valid, y_valid = process_train_data(
            X_train=self.X_train,
            y_train=self.y_train,
            X_valid=self.X_valid,
            y_valid=self.y_valid,
        )
        # fuzzifying X_train_new
        X_train_F = fuzzifying(
            fuzzy_type=self.fuzzy_type,
            fuzzy_cut=self.fuzzy_cut,
            X=X_train,
            number_of_intervals=self.number_of_intervals,
        )
        self.X_train_F = X_train_F
        # training and return rhs and lhs
        self.lhss, self.rhss = trained_model_for_X_y(
            smaller_better=self.smaller_better,
            X_train_F=self.X_train_F,
            X=X_train_F,
            y=y_train,
            metric=self.metric,
            threshold=self.threshold,
        )

        return self

    def predict(self, *args, **kwargs):
        """Predict function"""
        X_test = kwargs["X"]
        X_test_F = fuzzifying(
            fuzzy_type=self.fuzzy_type,
            fuzzy_cut=self.fuzzy_cut,
            X=X_test,
            number_of_intervals=self.number_of_intervals,
        )
        predictions = []
        y_paired_weights = pairwise_distances(X_test_F, self.lhss, metric=self.metric)
        index = 0
        for y_p in y_paired_weights:
            y_index = np.argmin(y_p, axis=0)
            predictions.append(self.rhss[y_index])
            index += 1
        return predictions
