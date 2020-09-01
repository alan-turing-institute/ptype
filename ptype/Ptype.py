# Mainly text manipulation utils
from ptype.utils import create_folders

# Latex, and visualization utils
from ptype.utils import print_to_file, save_object

import csv
import numpy as np
import os

from ptype.Config import Config
from ptype.Model import PtypeModel
from ptype.PFSMRunner import PFSMRunner
from scipy.stats import norm


class ColResult:
    def __init__(self, series):
        self.series = series
        self.p_t = {}
        self.predicted_type = None
        self.normal_values = []
        self.missing_values = []
        self.anomalous_values = []

    def get_unique_vals(self, return_counts=False):
        """ List of the unique values found in a column."""
        return np.unique([str(x) for x in self.series.tolist()], return_counts=return_counts)

    def has_missing(self):
        return self.missing_values != []

    def has_anomalous(self):
        return self.anomalous_values != []

    def show_results_for(self, indices, desc):
        if len(indices) == 0:
            count = 0
        else:
            unique_vals, unique_vals_counts = self.get_unique_vals(return_counts=True)
            vs = [unique_vals[ind] for ind in indices][:20]
            vs_counts = [unique_vals_counts[ind] for ind in indices][:20]
            count = sum(unique_vals_counts[indices])
            print('\t' + desc, vs)
            print('\ttheir counts: ', vs_counts)

        return count

    def show(self):
        print('col: ' + str(self.series.name))
        print('\tpredicted type: ' + self.predicted_type)
        print('\tposterior probs: ', self.p_t)

        normal = self.show_results_for(self.normal_values, "some normal data values: ")
        missing = self.show_results_for(self.missing_values, "missing values:")
        anomalies = self.show_results_for(self.anomalous_values, "anomalies:")

        total = normal + missing + anomalies

        print('\tfraction of normal:', round(normal / total, 2), '\n')
        print('\tfraction of missing:', round(missing / total, 2), '\n')
        print('\tfraction of anomalies:', round(anomalies / total, 2), '\n')

    def get_normal_predictions(self):
        """Values identified as 'normal'."""
        vs = self.get_unique_vals()
        return [vs[i] for i in self.normal_values]

    def get_missing_data_predictions(self):
        """Values identified as 'missing'."""
        vs = self.get_unique_vals()
        return [vs[i] for i in self.missing_values]

    def get_anomaly_predictions(self):
        """The values identified as 'anomalies'."""
        vs = self.get_unique_vals()
        return [vs[i] for i in self.anomalous_values]

    def remove_from_missing (self, indices):
        self.missing_values = list(set(self.missing_values) - set(indices))

    def remove_from_anomalies (self, indices):
        self.anomalous_values = list(set(self.anomalous_values) - set(indices))

    def add_to_normal (self, indices):
        self.normal_values = list(set(self.normal_values).union(set(indices)))

    def change_missing_data_annotations(self, missing_data):
        indices = [np.where(self.get_unique_vals() == v)[0][0] for v in missing_data]
        self.add_to_normal(indices)
        self.remove_from_missing(indices)

    def change_anomaly_annotations(self, anomalies):
        indices = [np.where(self.get_unique_vals() == v)[0][0] for v in anomalies]
        self.add_to_normal(indices)
        self.remove_from_anomalies(indices)

    def replace_missing(self, v):
        vs = self.get_unique_vals()
        for i in self.missing_values:
            self.series.replace(vs[i], v, inplace=True)


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
        self.verbose = False

    def set_data(self, df):
        _dataset_name = 'demo'
        df = df.applymap(str)
        # to refresh the outputs
        self.results = {} # column-indexed
        self.all_posteriors = {_dataset_name: {}}

        _column_names = df.columns

        # Creates a configuration object for the experiments
        config = Config(self.types, _dataset_name=_dataset_name, _column_names=_column_names)

        # Ptype model for inference
        if self.model is None:
            self.model = PtypeModel(config, df)
        else:
            self.model.set_params(config, df)

    ###################### MAIN METHODS #######################
    def run_inference(self, _data_frame):
        """ Runs inference for each column in a dataframe.
            The outputs are stored in dictionaries (see store_outputs).
            The column types are saved to a csv file.

        :param _data_frame:

        """

        self.set_data(_data_frame)

        if self.verbose:
            print_to_file('processing ' + self.model.experiment_config.dataset_name)

        # Normalizing the parameters to make sure they're probabilities
        self.normalize_params()

        # Generates a binary mask matrix to check if a word is supported by a PFSM or not. (this is just to optimize the implementation.)
        self.PFSMRunner.update_values(np.unique(self.model.data.values))

        # Calculate probabilities for each column, run inference and store results.
        for _, col in enumerate(list(self.model.experiment_config.column_names)):
            probabilities, counts = self.generate_probs_a_column(col)
            if self.verbose:
                print_to_file('\tinference is running...')
            self.model.run_inference(probabilities, counts)
            self.store_outputs(col)
            self.results[col] = self.column_results(col)

        # Export column types, and missing data
        save = False
        if save:
            self.write_type_predictions_2_csv(col.predicted_type for col in self.results.values())

    ####################### OUTPUT METHODS #########################
    def show_results_df(self):
        df_output = self.model.data.copy()
        df_output.columns = df_output.columns.map(
            lambda col: str(col) + '(' + self.results[col].predicted_type + ')')
        return df_output

    def show_results(self, cols=None):
        if cols is None:
            cols = self.results.keys()

        print('\ttypes: ', list(self.types.values()), '\n')

        for col in cols:
            self.results[col].show()

    def detect_missing_anomalies(self, inferred_column_type):
        normals, missings, anomalies = [], [], []
        if inferred_column_type != 'all identical':
            row_posteriors = self.model.p_z[:, np.argmax(self.model.p_t), :]
            max_row_posterior_indices = np.argmax(row_posteriors, axis=1)

            normals = list(np.where(max_row_posterior_indices == self.model.TYPE_INDEX)[0])
            missings = list(np.where(max_row_posterior_indices == self.model.MISSING_INDEX)[0])
            anomalies = list(np.where(max_row_posterior_indices == self.model.ANOMALIES_INDEX)[0])

        return [normals, missings, anomalies]

    def store_outputs(self, column_name):
        """ First stores the posterior distribution of the column type, and the predicted column type.
            Secondly, it stores the indices of the rows categorized according to the row types.

        :param column_name:
        """
        self.all_posteriors[self.model.experiment_config.dataset_name][column_name] = self.model.p_t

        # In case of a posterior vector whose entries are equal
        if len(set([i for i in self.model.p_t])) == 1:
            inferred_column_type = 'all identical'
        else:
            inferred_column_type = self.model.experiment_config.types_as_list[np.argmax(self.model.p_t)]

        # Indices for the unique values
        [normals, missings, anomalies] = self.detect_missing_anomalies(inferred_column_type)

    def column_results(self, col):
        """ First stores the posterior distribution of the column type, and the predicted column type.
            Secondly, it stores the indices of the rows categorized according to the row types.

        :param col:
        """
        result = ColResult(self.model.data[col])
        result.p_t = self.model.p_t

        # In case of a posterior vector whose entries are equal
        if len(set([i for i in self.model.p_t])) == 1:
            inferred_column_type = 'all identical'
        else:
            inferred_column_type = self.model.experiment_config.types_as_list[np.argmax(self.model.p_t)]
        result.predicted_type = inferred_column_type

        # Indices for the unique values
        [normals, missings, anomalies] = self.detect_missing_anomalies(inferred_column_type)
        result.normal_values = normals
        result.missing_values = missings
        result.anomalous_values = anomalies
        return result

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

    ####################### HELPERS #########################
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
        machines = self.PFSMRunner.machines
        for i, machine in enumerate(machines):

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

        self.PFSMRunner.machines = machines

    def initialize_params_randomly(self):
        LOG_EPS = -1e150

        # make uniform
        machines = self.PFSMRunner.machines
        for i, machine in enumerate(machines):

            # discards missing and anomaly types
            if i >= 2:
                # make uniform
                # machine.I = {a: np.log(np.random.uniform(0.,1.)) if machine.I[a] != LOG_EPS else LOG_EPS for a in machine.I}
                machine.I_z = {a: np.log(np.random.uniform(0., 1.)) if machine.I[a] != LOG_EPS else LOG_EPS for a in
                               machine.I}

                for a in machine.T:
                    for b in machine.T[a]:
                        for c in machine.T[a][b]:
                            machine.T_z[a][b][c] = np.log(np.random.uniform(0., 1.))

                machine.F_z = {a: np.log(np.random.uniform(0., 1.)) if machine.F[a] != LOG_EPS else LOG_EPS for a in
                               machine.F}

        self.PFSMRunner.machines = machines

    def generate_probs_a_column(self, column_name):
        """ Generates probabilities for the unique data values in a column.

        :param column_name: name of a column
        :return probabilities: an IxJ sized np array, where probabilities[i][j] is the probability generated for i^th unique value by the j^th PFSM.
                counts: an I sized np array, where counts[i] is the number of times i^th unique value is observed in a column.

        """
        unique_values_in_a_column, counts = self.get_unique_vals(column_name, return_counts=True)
        probabilities_dict = self.PFSMRunner.generate_machine_probabilities(unique_values_in_a_column)
        probabilities = np.array([probabilities_dict[str(x_i)] for x_i in unique_values_in_a_column])

        return probabilities, counts

    def normalize_params(self):
        for i, machine in enumerate(self.PFSMRunner.machines):
            if i not in [0, 1]:
                self.PFSMRunner.machines[i].I = self.model.normalize_initial(machine.I_z)
                self.PFSMRunner.machines[i].F, self.PFSMRunner.machines[i].T = self.model.normalize_final(machine.F_z,
                                                                                                          machine.T_z)

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

    def get_unique_vals(self, col, return_counts=False):
        """ List of the unique values found in a column."""
        return np.unique(
            [str(x) for x in self.model.data[col].tolist()],
            return_counts=return_counts
        )

    def get_normal_predictions(self, col):
        return self.results[col].get_normal_predictions()

    def get_missing_data_predictions(self, col):
        return self.results[col].get_missing_data_predictions()

    def get_anomaly_predictions(self, col):
        return self.results[col].get_anomaly_predictions()

    def get_columns_with_type(self, _type):
        return [col for col in self.results.keys() if self.results[col].predicted_type == _type]

    def get_columns_with_missing(self):
        return [col for col in self.results.keys() if self.results[col].has_missing()]

    def get_columns_with_anomalies(self):
        return [col for col in self.results.keys() if self.results[col].has_anomalous()]

    def change_column_type_annotations(self, cols, new_types):
        for col, new_type in zip(cols, new_types):
            print('The column type of ' + col + ' is changed from ' + self.results[col].predicted_type + ' to ' + new_type)
            self.results[col].predicted_type = new_type

    def change_missing_data_annotations(self, col, _missing_data):
        self.results[col].change_missing_data_annotations(_missing_data)

    def change_anomaly_annotations(self, col, anomalies):
        self.results[col].change_anomaly_annotations(anomalies)

    def replace_missing(self, col, v):
        self.results[col].replace_missing(v)
        self.run_inference(_data_frame=self.model.data)

    def remove_missing_and_anomalies(self, col, col_name):
        y = np.unique([str(int_element) for int_element in col.tolist()])
        entries_to_discard = self.results[col_name].missing_types + self.results[col_name].anomaly_types
        normal_entries = list(set(range(len(y))) - set(entries_to_discard))
        normal_data_values = y[normal_entries]

        return col.loc[col.isin(normal_data_values)]

    def get_categorical_columns(self):
        cats = {}
        for col_name in self.model.data.columns:
            col = self.model.data[col_name]
            col = self.remove_missing_and_anomalies(col, col_name)

            # just dropping certain values
            for encoding in ['NULL', 'null', 'Null', '#NA', '#N/A', 'NA', 'NA ', ' NA', 'N A', 'N/A', 'N/ A', 'N /A',
                             'N/A',
                             'na', ' na', 'na ', 'n a', 'n/a', 'N/O', 'NAN', 'NaN', 'nan', '-NaN', '-nan', '-', '!',
                             '?', '*', '.']:
                col = col.apply(lambda y: str(y).replace(encoding, ''))

            col = col.replace('', np.nan)
            col = col.dropna()
            col = list(col.values)

            res = self.get_categorical_signal_gaussian(col)
            if res[0]:
                cats[col_name] = [res[1], res[2]]
