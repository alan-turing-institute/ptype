from collections import OrderedDict
import numpy as np
import pandas as pd

from ptype.Column import Column, get_unique_vals
from ptype.Model import Model
from ptype.PFSMRunner import PFSMRunner
from ptype.utils import print_to_file, save_object


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
        df = df.applymap(str)
        self.cols = {}

        # Ptype model for inference
        if self.model is None:
            self.model = Model(self.types, df)
        else:
            self.model.set_params(self.types, df)

        # Normalize the parameters to make sure they're probabilities
        self.PFSMRunner.normalize_params()

        # Generate binary mask matrix to check if a word is supported by a PFSM or not (this is just to optimize the implementation)
        self.PFSMRunner.update_values(np.unique(self.model.data.values))

        # Calculate probabilities for each column and run inference.
        for _, col_name in enumerate(list(df.columns)):
            probabilities, counts = self.generate_probs(col_name)
            if self.verbose:
                print_to_file("\tinference is running...")
            self.model.run_inference(probabilities, counts)
            self.cols[col_name] = self.column(col_name, counts)

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
        ptype_pandas_mapping = {"integer": "Int64"}

        for col_name in df:
            new_dtype = ptype_pandas_mapping[schema[col_name].predicted_type]
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
        training_params = {
            "current_runner": self.PFSMRunner,
            "data_frames": data_frames,
            "labels": labels,
        }
        if self.model is None:
            self.model = Model(self.types, training_params=training_params)
        else:
            self.model.set_params(self.types, training_params=training_params)

        save_object(
            self.PFSMRunner, "models/training_runner_initial",
        )
        training_error = [self.calculate_total_error(data_frames, labels)]
        print(training_error)

        # Iterates over whole data points
        for it in range(_max_iter):
            if self.verbose:
                print_to_file("iteration = " + str(it))

            # Trains machines using all of the training data frames
            self.model.current_runner = self.update_PFSMs(self.PFSMRunner)

            # Calculate training and validation error at each iteration
            training_error.append(self.calculate_total_error(data_frames, labels))
            print(training_error)

            if it > 0:
                if training_error[-2] - training_error[-1] < 1e-4:
                    if self.verbose:
                        print_to_file("converged!")
                    save_object(
                        self.model.current_runner, "models/training_runner_final",
                    )
                    break

        save_object(training_error, "models/training_error")

    def update_PFSMs(self, runner):
        return self.model.update_PFSMs(runner)

    # OUTPUT METHODS #########################
    def show_schema(self):
        df = self.model.data.iloc[0:0, :].copy()
        df.loc[0] = [col.predicted_type for _, col in self.cols.items()]
        return df.rename(index={0: "type"})

    def show_missing_values(self):
        missing_values = {}
        for col_name in self.model.data:
            missing_values[col_name] = np.unique(
                self.cols[col_name].get_missing_values()
            )

        return pd.Series(missing_values)

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

    def detect_missing_anomalies(self, inferred_column_type):
        if inferred_column_type != "all identical":
            row_posteriors = self.model.p_z[
                :, self.types.index(max(self.model.p_t, key=self.model.p_t.get)), :
            ]
            max_row_posterior_indices = np.argmax(row_posteriors, axis=1)

            return [
                list(np.where(max_row_posterior_indices == self.model.TYPE_INDEX)[0]),
                list(
                    np.where(max_row_posterior_indices == self.model.MISSING_INDEX)[0]
                ),
                list(
                    np.where(max_row_posterior_indices == self.model.ANOMALIES_INDEX)[0]
                ),
            ]
        else:
            return [[], [], []]

    def column(self, col_name, counts):
        """ First stores the posterior distribution of the column type, and the predicted column type.
            Secondly, it stores the indices of the rows categorized according to the row types.
        """
        # In case of a posterior vector whose entries are equal
        if len(set(self.model.p_t.values())) == 1:
            predicted_type = "all identical"
        else:
            predicted_type = max(self.model.p_t, key=self.model.p_t.get)

        # Indices for the unique values
        [normals, missings, anomalies] = self.detect_missing_anomalies(predicted_type)

        return Column(
            series=self.model.data[col_name],
            counts=counts,
            p_t=self.model.p_t,
            predicted_type=predicted_type,
            p_z=self.model.p_z[
                :, self.types.index(max(self.model.p_t, key=self.model.p_t.get)), :
            ],  # need to handle the uniform case
            normal_values=normals,
            missing_values=missings,
            anomalous_values=anomalies,
        )

    def generate_probs(self, column_name):
        """ Generates probabilities for the unique data values in a column.

        :param column_name: name of a column
        :return probabilities: an IxJ sized np array, where probabilities[i][j] is the probability generated for i^th unique value by the j^th PFSM.
                counts: an I sized np array, where counts[i] is the number of times i^th unique value is observed in a column.

        """
        unique_values_in_a_column, counts = get_unique_vals(
            self.model.data[column_name], return_counts=True
        )
        probabilities_dict = self.PFSMRunner.generate_machine_probabilities(
            unique_values_in_a_column
        )
        probabilities = np.array(
            [probabilities_dict[str(x_i)] for x_i in unique_values_in_a_column]
        )

        return probabilities, counts

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

    def reclassify_column(self, col_name, new_t):
        if new_t not in self.types:
            print("Given type is unknown!")
        self.cols[col_name].predicted_type = new_t
        self.cols[col_name].p_t = OrderedDict(
            [(t, 1.0) if t == new_t else (t, 0.0) for t in self.types]
        )
        # update the arff types?
