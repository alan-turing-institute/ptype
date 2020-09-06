from ptype.utils import create_folders, print_to_file, save_object

import csv
from enum import Enum
import numpy as np

from ptype.Config import Config
from ptype.Model import PtypeModel
from ptype.PFSMRunner import PFSMRunner
from scipy.stats import norm


def get_unique_vals(col, return_counts=False):
    """List of the unique values found in a column."""
    return np.unique([str(x) for x in col.tolist()], return_counts=return_counts)


# Use same names and values as the constants in Model.py. Could consolidate.
class Status(Enum):
    TYPE = 1
    MISSING = 2
    ANOMALOUS = 3


class Column:
    def __init__(self, series):
        self.series = series
        self.p_t = {}
        self.p_z = {}
        self.predicted_type = None
        self.unique_vals = []
        self.unique_vals_counts = []
        self.unique_vals_status = []
        self.cache_unique_vals()

    def cache_unique_vals(self):
        """Call this to (re)initialise the cache of my unique values."""
        self.unique_vals, self.unique_vals_counts = get_unique_vals(self.series, return_counts=True)

    def has_missing(self):
        return self.get_missing_data_predictions() != []

    def has_anomalous(self):
        return self.get_anomaly_predictions() != []

    def show_results_for(self, status, desc):
        indices = [i for i, _ in enumerate(self.unique_vals) if self.unique_vals_status[i] == status]
        if len(indices) == 0:
            return 0
        else:
            print('\t' + desc, [self.unique_vals[i] for i in indices][:20])
            print('\ttheir counts: ', [self.unique_vals_counts[i] for i in indices][:20])
            return sum(self.unique_vals_counts[indices])

    def show_results(self):
        print('col: ' + str(self.series.name))
        print('\tpredicted type: ' + self.predicted_type)
        print('\tposterior probs: ', self.p_t)

        normal = self.show_results_for(Status.TYPE, "some normal data values: ")
        missing = self.show_results_for(Status.MISSING, "missing values:")
        anomalies = self.show_results_for(Status.ANOMALOUS, "anomalies:")

        total = normal + missing + anomalies

        print('\tfraction of normal:', round(normal / total, 2), '\n')
        print('\tfraction of missing:', round(missing / total, 2), '\n')
        print('\tfraction of anomalies:', round(anomalies / total, 2), '\n')

    def get_normal_predictions(self):
        """Values identified as 'normal'."""
        return [v for i, v in enumerate(self.unique_vals) if self.unique_vals_status[i] == Status.TYPE]

    def get_missing_data_predictions(self):
        return [v for i, v in enumerate(self.unique_vals) if self.unique_vals_status[i] == Status.MISSING]

    def get_anomaly_predictions(self):
        return [v for i, v in enumerate(self.unique_vals) if self.unique_vals_status[i] == Status.ANOMALOUS]

    # These can be combined now. What should this do when the supplied missing values aren't compatible
    # with the predicted type?
    def change_missing_data_annotations(self, missing_data):
        for i in [np.where(self.unique_vals == v)[0][0] for v in missing_data]:
            self.unique_vals_status[i] = Status.TYPE
#
    def change_anomaly_annotations(self, anomalies):
        for i in [np.where(self.unique_vals == v)[0][0] for v in anomalies]:
            self.unique_vals_status[i] = Status.TYPE
#
    def replace_missing(self, v):
        for i, u in enumerate(self.unique_vals):
            if self.unique_vals_status[i] == Status.MISSING:
                self.series.replace(u, v, inplace=True)
        self.cache_unique_vals()


class Ptype:
    avg_racket_time = None

    def __init__(self, _exp_num=0, _types=None):
        default_types = {1: 'integer', 2: 'string', 3: 'float', 4: 'boolean', 5: 'gender', 6: 'date-iso-8601',
                         7: 'date-eu', 8: 'date-non-std-subtype', 9: 'date-non-std'}
        self.exp_num = _exp_num
        self.types = default_types if _types is None else _types
        self.PFSMRunner = PFSMRunner(list(self.types.values()))
        self.model = None
        self.data_frames = None
        self.all_posteriors = {}
        self.features = {}
        self.verbose = False
        self.cols = {}  # column-indexed

    def set_data(self, df):
        _dataset_name = 'demo'
        df = df.applymap(str)
        self.cols = {}
        self.all_posteriors = {_dataset_name: {}}

        # Creates a configuration object for the experiments
        config = Config(self.types, _dataset_name=_dataset_name, _column_names=df.columns)

        # Ptype model for inference
        if self.model is None:
            self.model = PtypeModel(config, df)
        else:
            self.model.set_params(config, df)

    ###################### MAIN METHODS #######################
    def run_inference(self, _data_frame):
        """ Runs inference for each column in a dataframe.
            The outputs are stored in dictionaries.
            The column types are saved to a csv file.

        :param _data_frame:
        """
        self.set_data(_data_frame)

        if self.verbose:
            print_to_file('processing ' + self.model.experiment_config.dataset_name)

        # Normalize the parameters to make sure they're probabilities
        self.PFSMRunner.normalize_params()

        # Generate binary mask matrix to check if a word is supported by a PFSM or not (this is just to optimize the implementation)
        self.PFSMRunner.update_values(np.unique(self.model.data.values))

        # Calculate probabilities for each column, run inference and store results
        for _, col_name in enumerate(list(self.model.experiment_config.column_names)):
            probabilities, counts = self.generate_probs(col_name)
            if self.verbose:
                print_to_file('\tinference is running...')
            self.model.run_inference(probabilities, counts)
            self.all_posteriors[self.model.experiment_config.dataset_name][col_name] = self.model.p_t
            self.cols[col_name] = self.column_results(col_name)

            # Store additional features for canonical type inference
            self.store_features(col_name, counts)

        # Export column types, and missing data
        save = False
        if save:
            self.write_type_predictions_2_csv(col.predicted_type for col in self.cols.values())

    # OUTPUT METHODS #########################
    def show_results_df(self):
        df_output = self.model.data.copy()
        df_output.columns = df_output.columns.map(
            lambda col: str(col) + '(' + self.cols[col].predicted_type + ')')
        return df_output

    def detect_missing_anomalies(self, inferred_column_type):
        normals, missings, anomalies = [], [], []
        if inferred_column_type != 'all identical':
            row_posteriors = self.model.p_z[:, np.argmax(self.model.p_t), :]
            max_row_posterior_indices = np.argmax(row_posteriors, axis=1)

            normals = list(np.where(max_row_posterior_indices == self.model.TYPE_INDEX)[0])
            missings = list(np.where(max_row_posterior_indices == self.model.MISSING_INDEX)[0])
            anomalies = list(np.where(max_row_posterior_indices == self.model.ANOMALIES_INDEX)[0])

        return [normals, missings, anomalies]

    def column_results(self, col_name):
        """ First stores the posterior distribution of the column type, and the predicted column type.
            Secondly, it stores the indices of the rows categorized according to the row types.

         :param col_name:
        """
        col = Column(self.model.data[col_name])
        col.p_t = self.model.p_t

        # In case of a posterior vector whose entries are equal
        if len(set([i for i in self.model.p_t])) == 1:
            inferred_column_type = 'all identical'
        else:
            inferred_column_type = self.model.experiment_config.types_as_list[np.argmax(self.model.p_t)]
        col.predicted_type = inferred_column_type

        # need to handle the uniform case
        result.p_z = self.model.p_z[:,np.argmax(self.model.p_t),:]

        # Indices for the unique values
        [normals, missings, anomalies] = self.detect_missing_anomalies(inferred_column_type)
        col.normal_values = normals
        col.missing_values = missings
        col.anomalous_values = anomalies

        col.unique_vals_status = [None] * len(col.unique_vals)
        for i in normals:
            col.unique_vals_status[i] = Status.TYPE
        for i in missings:
            col.unique_vals_status[i] = Status.MISSING
        for i in anomalies:
            col.unique_vals_status[i] = Status.ANOMALOUS

        return col

    def store_features(self, col_name, counts):
        posterior = self.all_posteriors[self.model.experiment_config.dataset_name][col_name]

        sorted_posterior = [
            posterior[3],
            posterior[4:].sum(),
            posterior[2],
            posterior[0],
            posterior[1],
        ]

        entries = [
            str(int_element) for int_element in self.model.data[col_name].tolist()
        ]
        U = len(np.unique(entries))
        U_clean = len(self.cols[col_name].normal_values)

        N = len(entries)
        N_clean = sum([counts[index] for index in self.cols[col_name].normal_values])

        u_ratio = U / N
        if U_clean == 0 and N_clean == 0:
            u_ratio_clean = 0.0
        else:
            u_ratio_clean = U_clean / N_clean

        self.features[col_name] = np.array(
            sorted_posterior + [u_ratio, u_ratio_clean, U, U_clean,]
        )

    def save_posteriors(self, filename='all_posteriors.pkl'):
        save_object(self.all_posteriors, filename)

    def write_type_predictions_2_csv(self, column_type_predictions):
        with open(
                self.model.experiment_config.main_experiments_folder + '/type_predictions/' +
                self.model.experiment_config.dataset_name + '/type_predictions.csv',
                'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Column', 'F#', 'messytables', 'ptype', 'readr', 'Trifacta', 'hypoparsr'])
            for column_name, column_type_prediction in zip(self.model.experiment_config.column_names,
                                                           column_type_predictions):
                writer.writerow([column_name, '', '', column_type_prediction, '', '', ''])

    # HELPERS #########################
    def setup_a_column(self, i, column_name):
        if self.verbose:
            print_to_file('column # ' + str(i) + ' ' + column_name)

        # Sets parameters for folders of each column
        self.model.experiment_config.current_column = i
        self.model.experiment_config.current_column_name = column_name.replace(" ", "")
        self.model.experiment_config.current_experiment_folder = self.model.experiment_config.main_experiments_folder + '/' + self.model.experiment_config.dataset_name + '/' + self.model.experiment_config.current_column_name

        # Removes existing folders accordingly
        create_folders(self.model, i == 0)

    def initialize_params_uniformly(self):
        LOG_EPS = -1e150

        # make uniform
        for i, machine in enumerate(self.PFSMRunner.machines):
            # discards missing and anomaly types
            if i >= 2:
                # make uniform
                machine.I = {a: np.log(.5) if machine.I[a] != LOG_EPS else LOG_EPS for a in machine.I}
                machine.I_z = {a: np.log(.5) if machine.I[a] != LOG_EPS else LOG_EPS for a in machine.I}

                for a in machine.T:
                    for b in machine.T[a]:
                        for c in machine.T[a][b]:
                            machine.T[a][b][c] = np.log(.5)
                            machine.T_z[a][b][c] = np.log(.5)

                machine.F = {a: np.log(.5) if machine.F[a] != LOG_EPS else LOG_EPS for a in machine.F}
                machine.F_z = {a: np.log(.5) if machine.F[a] != LOG_EPS else LOG_EPS for a in machine.F}

    def generate_probs(self, column_name):
        """ Generates probabilities for the unique data values in a column.

        :param column_name: name of a column
        :return probabilities: an IxJ sized np array, where probabilities[i][j] is the probability generated for i^th unique value by the j^th PFSM.
                counts: an I sized np array, where counts[i] is the number of times i^th unique value is observed in a column.

        """
        unique_values_in_a_column, counts = get_unique_vals(self.model.data[column_name], return_counts=True)
        probabilities_dict = self.PFSMRunner.generate_machine_probabilities(unique_values_in_a_column)
        probabilities = np.array([probabilities_dict[str(x_i)] for x_i in unique_values_in_a_column])

        return probabilities, counts

    def calculate_error_df(self, dfs, labelss):
        # find the unique values in all of the columns once
        for i, df in enumerate(dfs):
            if i == 0:
                unique_vals = np.unique(df.values)
            else:
                unique_vals = np.concatenate((unique_vals, np.unique(df.values)))
        self.model.all_probs = self.PFSMRunner.generate_machine_probabilities(unique_vals)

        error = 0.

        for j, (df, labels) in enumerate(zip(dfs, labelss)):
            for i, column_name in enumerate(list(df.columns)):
                temp = self.model.f_col(str(j), column_name, labels[i] - 1)
                error += temp

                print(j, column_name, temp)

        return error

    def get_categorical_signal_gaussian(self, x, sigma=1, threshold=0.03):
        N = len(x)
        K = len(np.unique(x))

        if self.verbose:
            print(N, K, np.log(N))

        return [norm(np.log(N), np.log(N) / 2 * sigma).pdf(K) > threshold, np.log(N), K]

    def replace_missing(self, col, v):
        self.cols[col].replace_missing(v)
        self.run_inference(_data_frame=self.model.data)
