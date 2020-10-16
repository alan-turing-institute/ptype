import numpy as np

from ptype.Machine import (
    AnomalyNew,
    BooleansNew,
    Date_EUNewAuto,
    EmailAddress,
    FloatsNewAuto,
    Genders,
    IntegersNewAuto,
    IPAddress,
    ISO_8601NewAuto,
    MissingsNew,
    Nonstd_DateNewAuto,
    PI,
    StringsNewAuto,
    SubTypeNonstdDateNewAuto,
)

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


class Machines:
    def __init__(self, types):
        self.types = types
        self.forType = {t: MACHINES[t] for t in types}
        self.anomalous = AnomalyNew()
        self.missing = MissingsNew()
        self.normalize_params()

    @property
    def machines(self):
        return [self.missing, self.anomalous] + [self.forType[t] for t in self.forType]

    def machine_probabilities(self, col):
        return {
            str(v): [m.probability(str(v)) for m in self.machines] for v in col
        }

    def set_unique_values(self, unique_values):
        for machine in self.machines:
            machine.set_unique_values(unique_values)

    def remove_unique_values(self,):
        for machine in self.machines:
            machine.supported_words = {}

    def update_values(self, unique_values):
        self.remove_unique_values()
        self.set_unique_values(unique_values)

    def normalize_params(self):
        for machine in self.forType.values():
            machine.normalize_params()

    def initialize_uniformly(self):
        for machine in self.forType.values():
            machine.initialize_uniformly()

    def set_all_probabilities_z(self, w_j_z):
        counter = 0
        for machine in self.forType.values():
            counter = machine.set_probabilities_z(counter, w_j_z)

    def get_all_parameters_z(self):
        w_j = []
        for machine in self.forType.values():
            w_j.extend(machine.get_parameters_z())

        return w_j

    # fix magic number 0
    def set_na_values(self, na_values):
        self.missing.alphabet = na_values

    def get_na_values(self):
        return self.missing.alphabet.copy()

    # fix magic numbers 0, 1, 2
    def set_anomalous_values(self, anomalous_vals):

        probs = self.machine_probabilities(anomalous_vals)
        ratio = PI[0] / PI[2] + 0.1
        min_probs = {
            v: np.log(ratio * np.max(np.exp(probs[v]))) for v in anomalous_vals
        }

        self.anomalous.set_anomalous_values(anomalous_vals, min_probs)

    def get_anomalous_values(self):
        return self.anomalous.get_anomalous_values().copy()
