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
        return {
            str(v): [m.calculate_probability(str(v)) for m in self.machines] for v in col
        }

    def set_unique_values(self, unique_values):
        for _, machine in enumerate(self.machines):
            machine.set_unique_values(unique_values)

    def remove_unique_values(self,):
        for _, machine in enumerate(self.machines):
            machine.supported_words = {}

    def update_values(self, unique_values):
        self.remove_unique_values()
        self.set_unique_values(unique_values)

    def normalize_params(self):
        for _, machine in enumerate(self.machines[2:]):
            machine.normalize_params()

    def initialize_params_uniformly(self):
        for _, machine in enumerate(self.machines[2:]): # discards missing and anomaly types
            machine.initialize_params_uniformly()

    def set_all_probabilities_z(self, w_j_z):
        counter = 0
        for t, _ in enumerate(self.types):
            counter = self.machines[2 + t].set_probabilities_z(counter, w_j_z)

    def get_all_parameters_z(self):
        w_j = []
        for t, _ in enumerate(self.types):
            w_j.extend(self.machines[2 + t].get_parameters_z())

        return w_j
