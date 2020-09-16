import csv
import numpy as np
import pandas as pd

from ptype.Column import Column, Status, get_unique_vals
from ptype.Config import Config
from ptype.Model import PtypeModel
from ptype.PFSMRunner import PFSMRunner
from ptype.utils import create_folders, print_to_file, save_object


class Ptype:
    def __init__(self, _types=None):
        default_types = [
            "integer",
            "string",
            "float",
            "boolean",
            "gender",
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
    def fit_schema(self, _data_frame):
        """ Runs inference for each column in a dataframe, and returns a set of analysed columns.

        :param _data_frame:
        """
        _dataset_name = "demo"
        df = _data_frame.applymap(str)
        self.cols = {}

        # Creates a configuration object for the experiments
        config = Config(
            {i + 1: v for i, v in enumerate(self.types)}, _dataset_name=_dataset_name, _column_names=df.columns
        )

        # Ptype model for inference
        if self.model is None:
            self.model = PtypeModel(config, df)
        else:
            self.model.set_params(config, df)

        # Normalize the parameters to make sure they're probabilities
        self.PFSMRunner.normalize_params()

        # Generate binary mask matrix to check if a word is supported by a PFSM or not (this is just to optimize the implementation)
        self.PFSMRunner.update_values(np.unique(self.model.data.values))

        # Calculate probabilities for each column and run inference.
        for _, col_name in enumerate(list(self.model.config.column_names)):
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
         return self.update_dtypes(df.apply(self.as_normal(), axis=0), schema)

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

    def train_machines_multiple_dfs(
        self,
        data_frames,
        labels,
        _max_iter=20,
        _print=False,
        _test_data=None,
        _test_labels=None,
        _uniformly=False,
    ):
        """ Train the PFSMs given a set of dataframes and their labels

        :param labels: column types labeled by hand, where _label[i][j] denotes the type of j^th column in i^th dataframe.
        :param _max_iter: the maximum number of iterations the optimization algorithm runs as long as it's not converged.
        :param _print:
        :param _test_data:
        :param _test_labels:
        :param _uniformly: a binary variable used to initialize the PFSMs - True allows initializing uniformly rather than using hand-crafted values.
        :return:
        """
        self.print = _print

        if _uniformly:
            self.PFSMRunner.initialize_params_uniformly()

        if self.model is None:
            self.model = PtypeModel(config=None, data_frame=None)
        self.model.labels = labels
        self.model.types = {i + 1: v for i, v in enumerate(self.types)}

        # Setup folders and probabilities for all columns
        self.PFSMRunner.normalize_params()

        self.model.data_frames = data_frames

        # find the unique values in all of the columns once
        for i, df in enumerate(self.model.data_frames):
            if i == 0:
                unique_vals = np.unique(df.values)
            else:
                unique_vals = np.concatenate((unique_vals, np.unique(df.values)))
        self.model.unique_vals = unique_vals

        self.PFSMRunner.set_unique_values(unique_vals)

        # Finding unique values and their counts
        self.model.dfs_unique_vals_counts = {}
        for i, df in enumerate(data_frames):
            df_unique_vals_counts = {}
            for column_name in list(df.columns):
                temp_x, counts = np.unique(
                    [str(int_element) for int_element in df[column_name].tolist()],
                    return_counts=True,
                )
                counts = {u_data: c for u_data, c in zip(temp_x, counts)}
                temp_counts = list(counts.values())
                counts_array = np.reshape(temp_counts, newshape=(len(temp_counts),))
                df_unique_vals_counts[column_name] = [temp_x, counts_array]
            self.model.dfs_unique_vals_counts[str(i)] = df_unique_vals_counts

        # Setting
        self.model.J = len(
            self.PFSMRunner.machines
        )  # J: num of data types including missing and anomaly.
        self.model.K = (
            self.model.J - 2
        )  # K: num of possible column data types (excluding missing and anomaly)
        self.model.pi = [
            self.model.PI for j in range(self.model.K)
        ]  # mixture weights of row types
        self.model.current_runner = self.PFSMRunner

        training_error = [self.calculate_error_df(data_frames, labels)]

        save_object(
            self.PFSMRunner, "models/training_runner_initial",
        )
        print(training_error)

        # Iterates over whole data points
        for it in range(_max_iter):
            if self.verbose:
                print_to_file("iteration = " + str(it))

            # Trains machines using all of the training data frames
            self.PFSMRunner = self.train_all_models_multiple_dfs(self.PFSMRunner)
            self.model.current_runner = self.PFSMRunner

            # Calculate training and validation error at each iteration
            training_error.append(self.calculate_error_df(data_frames, labels))
            print(training_error)

            if it > 0:
                if training_error[-2] - training_error[-1] < 1e-4:
                    if self.verbose:
                        print_to_file("converged!")
                    save_object(
                        self.PFSMRunner, "models/training_runner_final",
                    )

                    break

        save_object(training_error, "models/training_error")

    def train_all_models_multiple_dfs(self, runner):
        if self.print:
            print_to_file("\ttraining is running...")
        return self.model.train_all_z_multiple_dfs_new(runner)

    # OUTPUT METHODS #########################
    def update_dtypes(self, df, schema):
        df_new = df.copy()

        ptype_pandas_mapping = {"integer": "Int64"}

        for col_name in self.model.data:
            new_dtype = ptype_pandas_mapping[schema[col_name].predicted_type]
            try:
                df_new[col_name] = df[col_name].astype(new_dtype)
            except TypeError:
                df_new[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype(
                    new_dtype
                )
        return df_new

    def show_schema(self):
        df = self.model.data.iloc[0:0, :].copy()
        df.loc[0] = [col.predicted_type for _, col in self.cols.items()]
        return df.rename(index={0: 'type'})

    def show_missing_values(self):
        missing_values = {}
        for col_name in self.model.data:
            missing_values[col_name] = np.unique(
                self.cols[col_name].get_missing_data_predictions()
            )

        return pd.Series(missing_values)

    def as_normal(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_normal_predictions() else pd.NA
        )

    def as_missing(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_missing_data_predictions() else pd.NA
        )

    def as_anomaly(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_anomaly_predictions() else pd.NA
        )

    def detect_missing_anomalies(self, inferred_column_type):
        if inferred_column_type != "all identical":
            row_posteriors = self.model.p_z[:, np.argmax(self.model.p_t), :]
            max_row_posterior_indices = np.argmax(row_posteriors, axis=1)

            return [
                list(np.where(max_row_posterior_indices == self.model.TYPE_INDEX)[0]),
                list(np.where(max_row_posterior_indices == self.model.MISSING_INDEX)[0]),
                list(np.where(max_row_posterior_indices == self.model.ANOMALIES_INDEX)[0]),
            ]
        else:
            return [[], [], []]

    def column(self, col_name, counts):
        """ First stores the posterior distribution of the column type, and the predicted column type.
            Secondly, it stores the indices of the rows categorized according to the row types.
        """
        # In case of a posterior vector whose entries are equal
        if len(set(self.model.p_t)) == 1:
            predicted_type = "all identical"
        else:
            predicted_type = self.model.config.types_as_list[np.argmax(self.model.p_t)]

        # Indices for the unique values
        [normals, missings, anomalies] = self.detect_missing_anomalies(
            predicted_type
        )

        return Column(
            series=self.model.data[col_name],
            counts=counts,
            p_t=self.model.p_t,
            predicted_type=predicted_type,
            p_z=self.model.p_z[:, np.argmax(self.model.p_t), :],  # need to handle the uniform case
            normal_values=normals,
            missing_values=missings,
            anomalous_values=anomalies
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

    def calculate_error_df(self, dfs, labelss):
        # find the unique values in all of the columns once
        for i, df in enumerate(dfs):
            if i == 0:
                unique_vals = np.unique(df.values)
            else:
                unique_vals = np.concatenate((unique_vals, np.unique(df.values)))
        self.model.all_probs = self.PFSMRunner.generate_machine_probabilities(
            unique_vals
        )

        error = 0.0

        for j, (df, labels) in enumerate(zip(dfs, labelss)):
            for i, column_name in enumerate(list(df.columns)):
                temp = self.model.f_col(str(j), column_name, labels[i] - 1)
                error += temp

        return error

    def replace_missing(self, col, v):
        self.cols[col].replace_missing(v)
        self.fit_schema(self.model.data)

    def reclassify_column(self, col_name, new_t):
        self.cols[col_name].predicted_type = new_t
        self.cols[col_name].p_t = [
            1.0 if t == new_t else 0.0 for t in self.model.config.types_as_list
        ]
        if new_t == "date":
            self.cols[col_name].p_t[5] = 1.0
        elif new_t not in self.model.config.types_as_list:
            print("Given type is unknown!")

        # update the arff types?
        # what if given type is not recognized

    # def reclassify_normal(self, col_name, vs):
    #     self.cols[col_name].reclassify_normal(vs)
    #     t_index = np.argmax(self.cols[col_name].p_t)
    #     t_index = [i if t==for i, t in enumerate(self.model.config.types_as_list)]
    #
    #     for i in [np.where(self.cols[col_name].unique_vals == v)[0][0] for v in vs]:
    #         self.cols[col_name].unique_vals_status[i] = Status.TYPE
