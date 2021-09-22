import joblib
import numpy as np

from pathlib import Path
from ptype import Ptype
from ptype.Column import Column
from ptype.Machines import Machines

resource_path = Path(__file__).parent / "../models"


class PtypeCat(Ptype.Ptype):
    """The ptype cat object."""

    def __init__(self):
        self.types = [
            "integer",
            "string",
            "float",
            "date-iso-8601",
            "date-eu",
            "date-non-std-subtype",
            "date-non-std",
        ]
        self.machines = Machines(self.types)
        self.verbose = False
        self.lr_clf = joblib.load(resource_path.joinpath("LR.sav"))
        self.scaler = joblib.load(resource_path.joinpath("scaler.pkl"))


    def _column(self, df, col_name, logP, counts):
        col = super()._column(df, col_name, logP, counts)

        # ptype-cat
        t_hat = col.inferred_type()
        if t_hat in ["integer", "string"]:
            feats = col._get_features(counts)
            # magic numbers
            feats[-2:] = self.scaler.transform(feats[-2:].reshape(1, -1))
            ind = np.where(self.lr_clf.classes_ == "categorical")[0][0]
            p_cat = self.lr_clf.predict_proba(feats.reshape(1, -1))[0][ind]
        else:
            p_cat = 0.0
        col.set_p_t_cat(t_hat, p_cat)

        return col

