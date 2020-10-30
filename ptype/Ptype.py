import numpy as np

from ptype.Column import (
    ANOMALIES_INDEX,
    MISSING_INDEX,
    TYPE_INDEX,
    Column,
    get_unique_vals,
)
from ptype.Machine import PI
from ptype.Machines import Machines
from ptype.Trainer import likelihoods_normalize, sum_weighted_likelihoods
from ptype.Schema import Schema
from ptype.utils import normalize_log_probs, LOG_EPS


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
        self.machines = Machines(self.types)
        self.verbose = False

    def schema_fit(self, df):
        """ Runs inference for each column in a dataframe, and returns a set of analysed columns.

        :param df:
        """
        df = df.applymap(str)  # really?
        self.machines.normalize_params()

        # Optimisation: generate binary mask matrix to check if words are supported by PFSMs
        self.machines.update_values(np.unique(df.values))

        # Calculate probabilities for each column and run inference.
        cols = {}
        for _, col_name in enumerate(list(df.columns)):
            unique_vs, counts = get_unique_vals(df[col_name], return_counts=True)
            probabilities_dict = self.machines.machine_probabilities(unique_vs)
            probabilities = np.array(
                [probabilities_dict[str(x_i)] for x_i in unique_vs]
            )

            cols[col_name] = self.column(df, col_name, probabilities, counts)

        return Schema(df, cols)

    def column(self, df, col_name, logP, counts):
        # Constants
        I, J = logP.shape  # num of rows x num of data types
        K = J - 2  # num of possible column data types (excluding missing and catch-all)

        # Inference
        p_t = []  # posterior probability distribution of column types
        p_z = {}  # posterior probability distribution of row types

        counts_array = np.array(counts)

        # Iterate for each possible column type
        for k in range(K):
            p_t.append(sum_weighted_likelihoods(counts_array, logP, k))
            x1, x2, x3, log_mx, sm = likelihoods_normalize(PI, logP, k)

            p_z_k = np.zeros((I, 3))
            p_z_k[:, TYPE_INDEX] = np.exp(x1 - log_mx - np.log(sm))
            p_z_k[:, MISSING_INDEX] = np.exp(x2 - log_mx - np.log(sm))
            p_z_k[:, ANOMALIES_INDEX] = np.exp(x3 - log_mx - np.log(sm))
            p_z[self.types[k]] = p_z_k / p_z_k.sum(axis=1)[:, np.newaxis]

        p_t = normalize_log_probs(p_t)

        return Column(
            series=df[col_name],
            counts=counts,
            p_t={t: p for t, p in zip(self.types, p_t)},
            p_z=p_z,
        )

    def get_na_values(self):
        return self.machines.missing.alphabet.copy()

    def set_na_values(self, na_values):
        self.machines.missing.alphabet = na_values

    def get_an_values(self):
        return self.machines.anomalous.an_values.copy()

    def set_an_values(self, an_values):
        probs = self.machines.machine_probabilities(an_values)

        # magic numbers!
        ratio = PI[0] / PI[2] + 0.1
        new_probs = {v: np.log(ratio * np.max(np.exp(probs[v]))) for v in an_values}

        self.machines.anomalous.set_an(an_values, new_probs)

    def get_string_alphabet(self):
        string_index = 2 + self.types.index("string")
        return self.machines.machines[string_index].alphabet

    def set_string_alphabet(self, alphabet):
        string_index = 2 + self.types.index("string")
        self.machines.machines[string_index].set_alphabet(alphabet)
