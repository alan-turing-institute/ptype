from copy import deepcopy
import numpy as np

from ptype.Column import ANOMALIES_INDEX, MISSING_INDEX, TYPE_INDEX, Column, get_unique_vals
from ptype.Model import LLHOOD_TYPE_START_INDEX, Model, PI
from ptype.PFSMRunner import PFSMRunner
from ptype.Schema import Schema
from ptype.utils import (
    log_weighted_sum_probs,
    log_weighted_sum_normalize_probs,
    normalize_log_probs
)

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

            cols[col_name] = self.run_inference(df, col_name, probabilities, counts)

        return Schema(df, cols)

    def run_inference(self, df, col_name, logP, counts):
        # Constants
        I, J = logP.shape   # num of rows x num of data types
        K = J - 2           # num of possible column data types (excluding missing and catch-all)

        # Initializations
        pi = [PI for k in range(K)]  # mixture weights of row types

        # Inference
        p_t = []            # posterior probability distribution of column types
        p_z = {}            # posterior probability distribution of row types

        counts_array = np.array(counts)

        # Iterate for each possible column type
        for k in range(K):

            # Sum of weighted likelihoods (log-domain)
            p_t.append(
                (
                    counts_array
                    * log_weighted_sum_probs(
                        pi[k][0],
                        logP[:, k + LLHOOD_TYPE_START_INDEX],
                        pi[k][1],
                        logP[:, MISSING_INDEX - 1],
                        pi[k][2],
                        logP[:, ANOMALIES_INDEX - 1],
                    )
                ).sum()
            )

            # Calculate posterior cell probabilities

            # Normalize
            x1, x2, x3, log_mx, sm = log_weighted_sum_normalize_probs(
                pi[k][0],
                logP[:, k + LLHOOD_TYPE_START_INDEX],
                pi[k][1],
                logP[:, MISSING_INDEX - 1],
                pi[k][2],
                logP[:, ANOMALIES_INDEX - 1],
            )

            p_z_k = np.zeros((I, 3))
            p_z_k[:, TYPE_INDEX] = np.exp(x1 - log_mx - np.log(sm))
            p_z_k[:, MISSING_INDEX] = np.exp(x2 - log_mx - np.log(sm))
            p_z_k[:, ANOMALIES_INDEX] = np.exp(x3 - log_mx - np.log(sm))
            p_z[self.types[k]] = p_z_k / p_z_k.sum(axis=1)[:, np.newaxis]

        p_t = normalize_log_probs(np.array(p_t))

        return Column(
            series=df[col_name],
            counts=counts,
            p_t={t: p for t, p in zip(self.types, p_t)},
            p_z=p_z
        )

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

    # fix magic numbers 0, 1, 2
    def set_anomalous_values(self, anomalous_vals):

        probs = self.PFSMRunner.generate_machine_probabilities(anomalous_vals)
        ratio = PI[0] / PI[2] + 0.1
        min_probs = {
            v: np.log(ratio * np.max(np.exp(probs[v]))) for v in anomalous_vals
        }

        self.PFSMRunner.machines[1].set_anomalous_values(anomalous_vals, min_probs)

    def get_anomalous_values(self):
        return self.PFSMRunner.machines[1].get_anomalous_values().copy()
