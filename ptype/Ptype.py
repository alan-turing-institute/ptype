from copy import deepcopy
import numpy as np
import pandas as pd

from ptype.Column import get_unique_vals
from ptype.Model import Model
from ptype.PFSMRunner import PFSMRunner
from ptype.utils import LOG_EPS


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
        self.cols = {}

    ###################### MAIN METHODS #######################
    def fit_schema(self, df):
        """ Runs inference for each column in a dataframe, and returns a set of analysed columns.

        :param df:
        """
        df = df.applymap(str)  # really?
        self.cols = {}
        self.model = Model(self.types, df)
        self.PFSMRunner.normalize_params()

        # Optimisation: generate binary mask matrix to check if words are supported by PFSMs
        self.PFSMRunner.update_values(np.unique(self.model.df.values))

        # Calculate probabilities for each column and run inference.
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

            # apply user feedback for missing data and anomalies
            # temporarily overwrite the proabilities for a given value and a column?
            self.cols[col_name] = self.model.run_inference(
                col_name, probabilities, counts
            )

        return self.cols

    def transform_schema(self, df, schema):
        """Transforms a data frame according to previously inferred schema.

         Parameters
         ----------
         df: Pandas dataframe object.

         Returns
         -------
         Transformed Pandas dataframe object.
         """
        df = df.apply(self.as_normal(), axis=0)
        ptype_pandas_mapping = {
            "integer": "Int64",
            "date-iso-8601": "datetime64",
            "date-eu": "datetime64",
            "date-non-std": "datetime64",
            "string": "string",
            "boolean": "boolean",  # will remove boolean later
            "float": "float64",
        }
        for col_name in df:
            new_dtype = ptype_pandas_mapping[schema[col_name].type]
            if new_dtype == "boolean":
                df[col_name] = df[col_name].apply(
                    lambda x: False
                    if str(x) in ["F"]
                    else (True if str(x) in ["T"] else x)
                )

            try:
                df[col_name] = df[col_name].astype(new_dtype)
            except TypeError:
                # TODO: explain why this case needed
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(
                    new_dtype
                )
        return df

    def fit_transform_schema(self, df):
        """Infers a schema and transforms a data frame accordingly.

        Parameters
        ----------
        df: Pandas dataframe object.

        Returns
        -------
        Transformed Pandas dataframe object.
        """
        return self.transform_schema(df, self.fit_schema(df))

    def train_model(
        self,
        data_frames,
        labels,
        _max_iter=20,
        _test_data=None,
        _test_labels=None,
        _uniformly=False,
    ):
        """ Train the PFSMs given a set of dataframes and their labels

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
        training_params = TrainingParams(self.PFSMRunner, data_frames, labels)
        assert self.model is None
        self.model = Model(self.types, training_params=training_params)

        initial = deepcopy(
            self.PFSMRunner
        )  # shouldn't need this, but too much mutation going on
        training_error = [self.calculate_total_error(data_frames, labels)]

        # Iterates over whole data points
        for n in range(_max_iter):
            # Trains machines using all of the training data frames
            self.model.update_PFSMs(self.PFSMRunner)
            self.model.current_runner = self.PFSMRunner  # why?

            # Calculate training and validation error at each iteration
            training_error.append(self.calculate_total_error(data_frames, labels))
            print(training_error)

            if n > 0:
                if training_error[-2] - training_error[-1] < 1e-4:
                    break

        return initial, self.model.current_runner, training_error

    # OUTPUT METHODS #########################
    def show_schema(self):
        df = self.model.df.iloc[0:0, :].copy()
        df.loc[0] = [col.type for _, col in self.cols.items()]
        df.loc[1] = [col.get_normal_values() for _, col in self.cols.items()]
        df.loc[2] = [col.get_normal_ratio() for _, col in self.cols.items()]
        df.loc[3] = [col.get_missing_values() for _, col in self.cols.items()]
        df.loc[4] = [col.get_missing_ratio() for _, col in self.cols.items()]
        df.loc[5] = [col.get_anomalous_values() for _, col in self.cols.items()]
        df.loc[6] = [col.get_anomalous_ratio() for _, col in self.cols.items()]
        return df.rename(
            index={
                0: "type",
                1: "normal values",
                2: "ratio of normal values",
                3: "missing values",
                4: "ratio of missing values",
                5: "anomalous values",
                6: "ratio of anomalous values",
            }
        )

    def as_normal(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_normal_values() else pd.NA
        )

    def as_missing(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_missing_values() else pd.NA
        )

    def as_anomaly(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_anomalous_values() else pd.NA
        )

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
