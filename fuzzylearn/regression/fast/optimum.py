from statistics import mode

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from fuzzylearn.regression.fast.fast import FLRegressor
from fuzzylearn.regression.fast.ray import FLRayRegressor
from fuzzylearn.util.read_data import read_yaml_file


class FLAutoOptunaRegressor:
    """FuzzyLearning class"""

    def __init__(self, *args, **kwargs):
        self.optimizer = kwargs["optimizer"]
        self.error_measurement_metric = kwargs["error_measurement_metric"]
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.smaller_better = True
        self.features = None
        self.y_for_color_code = None
        self.model = None
        try:
            self.n_trials = kwargs["n_trials"]
        except Exception as e:
            self.n_trials = None

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
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def error_measurement_metric(self):
        return self._error_measurement_metric

    @error_measurement_metric.setter
    def error_measurement_metric(self, value):
        self._error_measurement_metric = value

    def _optimizer_func(self, *args, **kwargs):
        # Set the random seed
        seed = 42
        np.random.seed(seed)

        # Get the number of rows in each array
        if self.X_valid is None or self.y_valid is None:
            if isinstance(self.X_train, np.ndarray):
                (
                    self.X_valid,
                    X_valid_test,
                    self.y_valid,
                    y_valid_test,
                ) = train_test_split(self.X_train, self.y_train, test_size=0.5)

            if isinstance(self.X_train, pd.DataFrame):
                (
                    self.X_valid,
                    X_valid_test,
                    self.y_valid,
                    y_valid_test,
                ) = train_test_split(self.X_train, self.y_train, test_size=0.5)

        # Path to your YAML file
        file_path = "fuzzylearn/optimization_conf.yaml"

        # Read the YAML file
        with open(file_path, "r") as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print("Error loading YAML file:", e)
        if self.optimizer == "auto_optuna" or self.optimizer == "auto_optuna_ray":
            for mtr in yaml_data["metrics_for_regression"]:
                str1 = set(
                    self.error_measurement_metric.lower()
                )  # Convert characters to lowercase and create a set
                str2 = set(mtr.lower())
                common_characters = str1.intersection(str2)
                if len(common_characters) >= 0.9 * len(str1):
                    metrics_for_regression = mtr
                    break
            if metrics_for_regression is None:
                raise ValueError(
                    f"{self.error_measurement_metric} is not acceptable! use something like mean_absolute_error(y_true, y_pred)"
                )

        def objective(trial):
            metric = yaml_data["metric"]
            fuzzy_cut = yaml_data["fuzzy_cut"]
            fuzzy_type = yaml_data["fuzzy_type"]
            number_of_intervals = yaml_data["number_of_intervals"]
            threshold = yaml_data["threshold"]
            # Define selection parameter
            metric = trial.suggest_categorical("metric", metric)
            number_of_intervals = trial.suggest_int(
                "number_of_intervals",
                int(number_of_intervals[0]),
                int(number_of_intervals[len(number_of_intervals) - 1]),
            )
            threshold = trial.suggest_float(
                "threshold", float(threshold[0]), float(threshold[len(threshold) - 1])
            )
            fuzzy_type = trial.suggest_categorical("fuzzy_type", fuzzy_type)
            if fuzzy_type == "triangular":
                fuzzy_cut = trial.suggest_float(
                    "fuzzy_cut",
                    float(fuzzy_cut[0]),
                    float(fuzzy_cut[len(fuzzy_cut) - 1]),
                )
            params = {
                "metric": metric,
                "number_of_intervals": number_of_intervals,
                "threshold": threshold,
                "fuzzy_type": fuzzy_type,
                "fuzzy_cut": fuzzy_cut,
            }

            if self.optimizer == "auto_optuna":
                self.model = FLRegressor(**params)
            if self.optimizer == "auto_optuna_ray":
                self.model = FLRayRegressor(**params)
            self.model.fit(X=self.X_valid, y=self.y_valid, X_valid=None, y_valid=None)
            y_pred = self.model.predict(X=X_valid_test)
            y_true = y_valid_test
            # Calculate accuracy score as the objective
            score = eval(metrics_for_regression)
            if trial.number > 1000 or score < 0.05:
                # Stop the study if the results are already good enough
                study.stop()
                if self.optimizer == "auto_optuna":
                    self.model = FLRegressor(**study.best_params)
                if self.optimizer == "auto_optuna_ray":
                    self.model = FLRayRegressor(**study.best_params)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=None)
        self.model = FLRegressor(**study.best_params)
        print("study.best_params", study.best_params)
        return self.model

    def fit(self, *args, **kwargs):
        """Fit function."""

        # arguments for the fit
        self.X_valid = kwargs["X_valid"]
        self.y_valid = kwargs["y_valid"]
        self.X_train = kwargs["X"]
        self.y_train = kwargs["y"]
        if self.optimizer == "auto_optuna":
            self.model = self._optimizer_func(optimizer="auto_optuna")
            print("this is best model using Optuna", self.model)
        if self.optimizer == "auto_optuna_ray":
            self.model = self._optimizer_func(optimizer="auto_optuna_ray")
            print("this is best model using Optuna and Ray", self.model)

        self.model.fit(*args, **kwargs)

        return self

    def predict(self, *args, **kwargs):
        """Predict function"""
        predictions = self.model.predict(*args, **kwargs)
        return predictions

    def feature_improtance(self, *args, **kwargs):
        """Feature improtance"""
        from collections import OrderedDict

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 8))

        lhss = self.lhss
        # Get the number of columns in the array
        num_columns = lhss.shape[1]

        # Generate the x-axis values for the bars
        x_values = np.arange(num_columns)
        x_values = self.features
        # Example list of numbers to determine colors
        colors = list(set(self.y_for_color_code.iloc[:, 0].to_list()))
        colors = [float(x) for x in colors]

        # Create a color map
        cmap = plt.cm.get_cmap("viridis")  # Choose a colormap, such as 'cool'

        # Create lines connecting the bars
        for i in range(lhss.shape[0] - 1):
            plt.plot(
                lhss[i, :],
                x_values,
                linestyle="-",
                color=cmap(colors[int(self.y_for_color_code.iloc[i, :].to_list()[0])]),
                label=str(int(self.y_for_color_code.iloc[i, :].to_list()[0])),
            )

        plt.legend(self.y_for_color_code)

        # Set labels and a title for the plot
        plt.xlabel("Levels")
        plt.ylabel("Features/Variables")
        plt.title("Fuzzified Features Map")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))

        # Display a legend for the rows
        plt.legend(by_label.values(), by_label.keys())

        # Show the plot
        plt.show()


class FLOptunaRegressor:
    """FuzzyLearning class"""

    def __init__(self, *args, **kwargs):
        self.optimizer = kwargs["optimizer"]
        self.error_measurement_metric = kwargs["error_measurement_metric"]
        self.number_of_intervals_range = kwargs["number_of_intervals_range"]
        self.threshold_range = kwargs["threshold_range"]
        self.metrics_list = kwargs["metrics_list"]
        self.fuzzy_type_list = kwargs["fuzzy_type_list"]
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.smaller_better = True
        self.features = None
        self.y_for_color_code = None
        self.model = None
        try:
            self.n_trials = kwargs["n_trials"]
        except Exception as e:
            self.n_trials = None
        try:
            self.fuzzy_cut_range = kwargs["fuzzy_cut_range"]
        except Exception as e:
            self.fuzzy_cut_range = None

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
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def error_measurement_metric(self):
        return self._error_measurement_metric

    @error_measurement_metric.setter
    def error_measurement_metric(self, value):
        self._error_measurement_metric = value

    @property
    def number_of_intervals_range(self):
        return self._number_of_intervals_range

    @number_of_intervals_range.setter
    def number_of_intervals_range(self, value):
        self._number_of_intervals_range = value

    @property
    def metrics_list(self):
        return self._metrics_list

    @metrics_list.setter
    def metrics_list(self, value):
        self._metrics_list = value

    @property
    def threshold_range(self):
        return self._threshold_range

    @threshold_range.setter
    def threshold_range(self, value):
        self._threshold_range = value

    @property
    def fuzzy_type_list(self):
        return self._fuzzy_type_list

    @fuzzy_type_list.setter
    def fuzzy_type_list(self, value):
        self._fuzzy_type_list = value

    @property
    def fuzzy_cut_range(self):
        return self._fuzzy_cut_range

    @fuzzy_cut_range.setter
    def fuzzy_cut_range(self, value):
        self._fuzzy_cut_range = value

    def _optimizer_func(self, *args, **kwargs):
        # Set the random seed
        seed = 42
        np.random.seed(seed)

        # Get the number of rows in each array
        if self.X_valid is None or self.y_valid is None:
            num_rows = self.X_train.shape[0]

            if isinstance(self.X_train, np.ndarray):
                (
                    self.X_valid,
                    X_valid_test,
                    self.y_valid,
                    y_valid_test,
                ) = train_test_split(self.X_train, self.y_train, test_size=0.5)

            elif isinstance(self.X_train, pd.DataFrame):
                (
                    self.X_valid,
                    X_valid_test,
                    self.y_valid,
                    y_valid_test,
                ) = train_test_split(self.X_train, self.y_train, test_size=0.5)
            else:
                raise ValueError(
                    f"this type of data {type(self.X_train)} is not supported for X_valid !!!"
                )

        # Path to your YAML file
        file_path = "fuzzylearn/optimization_conf.yaml"

        # Read the YAML file
        with open(file_path, "r") as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print("Error loading YAML file:", e)
        if self.optimizer == "optuna" or self.optimizer == "optuna_ray":
            for mtr in yaml_data["metrics_for_regression"]:
                str1 = set(
                    self.error_measurement_metric.lower()
                )  # Convert characters to lowercase and create a set
                str2 = set(mtr.lower())
                common_characters = str1.intersection(str2)
                if len(common_characters) >= 0.9 * len(str1):
                    metrics_for_regression = mtr
                    break
            if metrics_for_regression is None:
                raise ValueError(
                    f'{self.error_measurement_metric} is not acceptable! use something like f1_score(y_true, y_pred, average="weighted")'
                )

        def objective(trial):
            metric = self.metrics_list
            number_of_intervals = self.number_of_intervals_range
            threshold = self.threshold_range
            fuzzy_type = self.fuzzy_type_list
            fuzzy_cut = self.fuzzy_cut_range
            # Define selection parameter
            metric = trial.suggest_categorical("metric", metric)
            number_of_intervals = trial.suggest_int(
                "number_of_intervals",
                number_of_intervals[0],
                number_of_intervals[len(number_of_intervals) - 1],
            )
            threshold = trial.suggest_float(
                "threshold", threshold[0], threshold[len(threshold) - 1]
            )
            fuzzy_type = trial.suggest_categorical("fuzzy_type", fuzzy_type)
            if fuzzy_type == "triangular":
                fuzzy_cut = trial.suggest_float(
                    "fuzzy_cut",
                    float(fuzzy_cut[0]),
                    float(fuzzy_cut[len(fuzzy_cut) - 1]),
                )

            params = {
                "metric": metric,
                "number_of_intervals": number_of_intervals,
                "threshold": threshold,
                "fuzzy_type": fuzzy_type,
                "fuzzy_cut": fuzzy_cut,
            }
            if self.optimizer == "optuna":
                self.model = FLRegressor(**params)
            if self.optimizer == "optuna_ray":
                self.model = FLRayRegressor(**params)

            self.model.fit(X=self.X_valid, y=self.y_valid, X_valid=None, y_valid=None)
            y_pred = self.model.predict(X=X_valid_test)
            y_true = y_valid_test
            # Calculate accuracy score as the objective
            score = eval(metrics_for_regression)
            if trial.number > 1000 or score < 0.05:
                # Stop the study if the results are already good enough
                study.stop()
                if self.optimizer == "optuna":
                    self.model = FLRegressor(**study.best_params)
                if self.optimizer == "optuna_ray":
                    self.model = FLRayRegressor(**study.best_params)

            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.model = FLRegressor(**study.best_params)
        print("study.best_params", study.best_params)
        return self.model

    def fit(self, *args, **kwargs):
        """Fit function."""

        # arguments for the fit
        self.X_valid = kwargs["X_valid"]
        self.y_valid = kwargs["y_valid"]
        self.X_train = kwargs["X"]
        self.y_train = kwargs["y"]
        if self.optimizer == "optuna":
            self.model = self._optimizer_func(optimizer="auto_optuna")
            print("this is best model", self.model)
        if self.optimizer == "optuna_ray":
            self.model = self._optimizer_func(optimizer="auto_optuna_ray")
            print("this is best model using Optuna and Ray", self.model)

        self.model.fit(*args, **kwargs)

        return self

    def predict(self, *args, **kwargs):
        """Predict function"""
        predictions = self.model.predict(*args, **kwargs)
        return predictions

    def feature_improtance(self, *args, **kwargs):
        """Feature improtance"""
        from collections import OrderedDict

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 8))

        lhss = self.lhss
        # Get the number of columns in the array
        num_columns = lhss.shape[1]

        # Generate the x-axis values for the bars
        x_values = np.arange(num_columns)
        x_values = self.features
        # Example list of numbers to determine colors
        colors = list(set(self.y_for_color_code.iloc[:, 0].to_list()))
        colors = [float(x) for x in colors]

        # Create a color map
        cmap = plt.cm.get_cmap("viridis")  # Choose a colormap, such as 'cool'

        # Create lines connecting the bars
        for i in range(lhss.shape[0] - 1):
            plt.plot(
                lhss[i, :],
                x_values,
                linestyle="-",
                color=cmap(colors[int(self.y_for_color_code.iloc[i, :].to_list()[0])]),
                label=str(int(self.y_for_color_code.iloc[i, :].to_list()[0])),
            )

        plt.legend(self.y_for_color_code)

        # Set labels and a title for the plot
        plt.xlabel("Levels")
        plt.ylabel("Features/Variables")
        plt.title("Fuzzified Features Map")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))

        # Display a legend for the rows
        plt.legend(by_label.values(), by_label.keys())

        # Show the plot
        plt.show()
