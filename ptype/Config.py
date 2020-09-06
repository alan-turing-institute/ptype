class Config:
    # helps to store settings for an experiment.
    def __init__(
        self, _types, _dataset_name="dataset", _column_names="unknown",
    ):

        self.dataset_name = _dataset_name
        self.column_names = _column_names
        self.types = _types
        self.types_as_list = list(_types.values())

        columns = [
            "missing",
            "anomaly",
        ]
        for key in _types:
            columns.append(_types[key])
        self.columns = columns
