import sys
import numpy as np

from ptype.Machine import (
    IntegersNewAuto,
    StringsNewAuto,
    AnomalyNew,
    FloatsNewAuto,
    MissingsNew,
    BooleansNew,
    Genders,
    ISO_8601NewAuto,
    Date_EUNewAuto,
    Nonstd_DateNewAuto,
    SubTypeNonstdDateNewAuto,
    IPAddress,
    EmailAddress,
)
from ptype.utils import contains_all
from ptype.Model import Model
from ptype.Model import LOG_EPS

MACHINES = {
    "integer": IntegersNewAuto(),
    "string": StringsNewAuto(),
    "float": FloatsNewAuto(),
    "boolean": BooleansNew(),
    "gender": Genders(),
    "date-iso-8601": ISO_8601NewAuto(),
    "date-eu": Date_EUNewAuto(),
    "date-non-std-subtype": SubTypeNonstdDateNewAuto(),
    "date-non-std": Nonstd_DateNewAuto(),
    "IPAddress": IPAddress(),
    "EmailAddress": EmailAddress(),
}


class PFSMRunner:
    def __init__(self, types):
        self.types = types
        self.machines = [MissingsNew(), AnomalyNew()] + [MACHINES[t] for t in types]
        self.normalize_params()

    def generate_machine_probabilities(self, col):
        """ generates automata probabilities for a given column of data
        """
        return {
            str(v): [m.calculate_probability(str(v)) for m in self.machines] for v in col
        }

    def set_unique_values(self, unique_values):
        for i, machine in enumerate(self.machines):

            machine.supported_words = {}

            for unique_value in unique_values:
                if contains_all(unique_value, machine.alphabet):
                    machine.supported_words[unique_value] = 1
                else:
                    machine.supported_words[unique_value] = 0

            self.machines[i].supported_words = machine.supported_words

    def remove_unique_values(self,):
        for i, machine in enumerate(self.machines):
            self.machines[i].supported_words = {}

    def update_values(self, unique_values):
        self.remove_unique_values()
        self.set_unique_values(unique_values)

    def normalize_params(self):
        for i, machine in enumerate(self.machines[2:]):
            machine.I = Model.normalize_initial(machine.I_z)
            machine.F, machine.T = Model.normalize_final(machine.F_z, machine.T_z)

    def initialize_params_uniformly(self):
        for i, machine in enumerate(self.machines[2:]): # discards missing and anomaly types
            machine.I = {
                a: np.log(0.5) if machine.I[a] != LOG_EPS else LOG_EPS
                for a in machine.I
            }
            machine.I_z = {
                a: np.log(0.5) if machine.I[a] != LOG_EPS else LOG_EPS
                for a in machine.I
            }

            for a in machine.T:
                for b in machine.T[a]:
                    for c in machine.T[a][b]:
                        machine.T[a][b][c] = np.log(0.5)
                        machine.T_z[a][b][c] = np.log(0.5)

            machine.F = {
                a: np.log(0.5) if machine.F[a] != LOG_EPS else LOG_EPS
                for a in machine.F
            }
            machine.F_z = {
                a: np.log(0.5) if machine.F[a] != LOG_EPS else LOG_EPS
                for a in machine.F
            }

    def set_all_probabilities_z(self, w_j_z, normalize=False) -> object:
        counter = 0
        for t, _ in enumerate(self.types):
            machine = self.machines[2 + t]
            for state in machine.I:
                if machine.I[state] != LOG_EPS:
                    machine.I_z[state] = w_j_z[counter]
                    counter += 1

            for a in machine.T:
                for b in machine.T[a]:
                    for c in machine.T[a][b]:
                        machine.T_z[a][b][c] = w_j_z[counter]
                        counter += 1

            for state in machine.F:
                if machine.F[state] != LOG_EPS:
                    machine.F_z[state] = w_j_z[counter]
                    counter += 1

                if normalize:
                    machine.F_z, machine.T_z = Model.normalize_a_state(machine.F_z, machine.T_z, state)
                    machine.F, machine.T = machine.F_z, machine.T_z
                    machine.I_z = Model.normalize_initial(machine.I_z)
                    machine.I = machine.I_z

        return self

    def get_all_parameters_z(self):
        w_j = []
        for t, _ in enumerate(self.types):
            machine = self.machines[2 + t]

            w_j.extend([p for p in machine.I.values() if p != LOG_EPS])
            w_j.extend([p for a in machine.T_z.values() for b in a.values() for p in b.values()])
            w_j.extend([p for p in machine.F.values() if p != LOG_EPS])

        return w_j
