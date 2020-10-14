from copy import deepcopy
import numpy as np
import pandas as pd

from ptype.Column import get_unique_vals
from ptype.Model import Model
from ptype.PFSMRunner import PFSMRunner
from ptype.Schema import Schema


class TrainingParams:
    def __init__(self, current_runner, dfs, labels):
        self.current_runner = current_runner
        self.dfs = dfs
        self.labels = labels


class Ptype:
    def __init__(self, _types=None):
        default_types = [
            "integer",
            "string",
            "float",
            "boolean",
            "date-iso-8601",
            "date-eu",
            "date-non-std-subtype",
            "date-non-std",
        ]
        self.types = default_types if _types is None else _types
        self.PFSMRunner = PFSMRunner(self.types)
        self.model = None
        self.verbose = False

    def schema_fit(self, df):
        """ Runs inference for each column in a dataframe, and returns a set of analysed columns.

        :param df:
        """
        df = df.applymap(str)  # really?
        self.model = Model(self.types, df)
        self.PFSMRunner.normalize_params()

        # Optimisation: generate binary mask matrix to check if words are supported by PFSMs
        self.PFSMRunner.update_values(np.unique(self.model.df.values))

        # Calculate probabilities for each column and run inference.
        cols = {}
        for _, col_name in enumerate(list(df.columns)):
            unique_vs, counts = get_unique_vals(
                self.model.df[col_name], return_counts=True
            )
            probabilities_dict = self.PFSMRunner.generate_machine_probabilities(
                unique_vs
            )
            probabilities = np.array(
                [probabilities_dict[str(x_i)] for x_i in unique_vs]
            )

            cols[col_name] = self.model.run_inference(
                col_name, probabilities, counts
            )

        return Schema(df, cols)

    def train_model(
        self,
        dfs,
        labels,
        _max_iter=20,
        _test_data=None,
        _test_labels=None,
        _uniformly=False,
    ):
        """ Train the PFSMs given a set of dataframes and their labels

        :param dfs: data frames to train with.
        :param labels: column types labeled by hand, where _label[i][j] denotes the type of j^th column in i^th dataframe.
        :param _max_iter: the maximum number of iterations the optimization algorithm runs as long as it's not converged.
        :param _test_data:
        :param _test_labels:
        :param _uniformly: a binary variable used to initialize the PFSMs - True allows initializing uniformly rather than using hand-crafted values.
        :return:
        """
        if _uniformly:
            self.PFSMRunner.initialize_params_uniformly()
            self.PFSMRunner.normalize_params()

        # Ptype model for training
        training_params = TrainingParams(self.PFSMRunner, dfs, labels)
        assert self.model is None
        self.model = Model(self.types, training_params=training_params)

        initial = deepcopy(self.PFSMRunner)  # shouldn't need this, but too much mutation going on
        training_error = [self.calculate_total_error(dfs, labels)]

        # Iterates over whole data points
        for n in range(_max_iter):
            # Trains machines using all of the training data frames
            self.model.update_PFSMs(self.PFSMRunner)
            self.model.current_runner = self.PFSMRunner  # why?

            # Calculate training and validation error at each iteration
            training_error.append(self.calculate_total_error(dfs, labels))
            print(training_error)

            if n > 0:
                if training_error[-2] - training_error[-1] < 1e-4:
                    break

        return initial, self.model.current_runner, training_error

    def calculate_total_error(self, dfs, labels):
        self.model.all_probs = self.model.current_runner.generate_machine_probabilities(
            self.model.unique_vals
        )

        error = 0.0
        for j, (df, df_labels) in enumerate(zip(dfs, labels)):
            for i, column_name in enumerate(list(df.columns)):
                temp = self.model.f_col(str(j), column_name, df_labels[i] - 1)
                error += temp

        return error

    # fix magic number 0
    def set_na_values(self, na_values):
        self.PFSMRunner.machines[0].alphabet = na_values

    def get_na_values(self):
        return self.PFSMRunner.machines[0].alphabet.copy()

    # fix magic numbers 1, self.model.PI[0]+1e-10
    def set_anomalous_values(self, anomalous_vals):

        probs = self.PFSMRunner.generate_machine_probabilities(anomalous_vals)
        ratio = self.model.PI[0] / self.model.PI[2] + 0.1
        min_probs = {
            v: np.log(ratio * np.max(np.exp(probs[v]))) for v in anomalous_vals
        }

        self.PFSMRunner.machines[1].set_anomalous_values(anomalous_vals, min_probs)

    def get_anomalous_values(self):
        return self.PFSMRunner.machines[1].get_anomalous_values().copy()
