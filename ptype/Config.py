class Config:
    # helps to store settings for an experiment.
    def __init__(self, _types, _experiments_folder_path='experiments', _dataset_name='dataset', _column_names='unknown'):

        self.main_experiments_folder = _experiments_folder_path
        self.dataset_name = _dataset_name
        self.column_names = _column_names
        self.types = _types
        self.types_as_list = list(_types.values())

        columns = ['missing', 'catch-all',]
        for key in _types:
            columns.append(_types[key])
        self.columns = columns
