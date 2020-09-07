import clevercsv as csv
import json
import pandas as pd
from ptype.Ptype import Ptype
from tests.utils import get_datasets, evaluate_predictions


def read_data(_data_path, dataset_name):
    # wrong encoding leads to additional characters in the dataframe columns
    if dataset_name in ['mass_6.csv']:
        encoding = 'ISO-8859-1'
    else:
        encoding = 'utf-8'
    return csv.csv2df(_data_path + dataset_name, encoding=encoding, dtype=str, skipinitialspace=True)


def as_normal(ptype):
    return lambda series: \
        series.map(lambda v: v if v in ptype.cols[series.name].get_normal_predictions() else pd.NA)


def as_missing(ptype):
    return lambda series: \
        series.map(lambda v: v if v in ptype.cols[series.name].get_missing_data_predictions() else pd.NA)


def as_anomaly(ptype):
    return lambda series: \
        series.map(lambda v: v if v in ptype.cols[series.name].get_anomaly_predictions() else pd.NA)


def get_predictions(_data_path, dataset_name):
    # create ptype
    ptype = Ptype(_types=
                  {1: 'integer', 2: 'string', 3: 'float', 4: 'boolean',
                   5: 'date-iso-8601', 6: 'date-eu', 7: 'date-non-std-subtype',
                   8: 'date-non-std'})

    df = read_data(_data_path, dataset_name)
    ptype.run_inference(_data_frame=df)

    df_missing = df.apply(as_missing(ptype), axis=0)
    df_anomaly = df.apply(as_anomaly(ptype), axis=0)
    df_normal = df.apply(as_normal(ptype), axis=0)
    print(dataset_name)
    print('Original data:\n', df)
    print('Missing data:\n', df_missing)
    print('Anomalies:\n', df_anomaly)
    print('Normal:\n', df_normal)

    return {col_name: col.predicted_type for col_name, col in ptype.cols.items()}


def notebook_tests():
    import os
    if os.system("pytest --nbval notebooks/*.ipynb") != 0:
        raise Exception("Notebook test(s) failed.")


def main():
    data_folder = 'data/'
    annotations_file = 'annotations/annotations.json'
    predictions_file = 'tests/column_type_predictions.json'

    annotations = json.load(open(annotations_file))

    type_predictions = {}
    for dataset_name in get_datasets():
        type_predictions[dataset_name] = get_predictions(data_folder, dataset_name)

    with open(predictions_file, 'r', encoding='utf-8-sig') as read_file:
        expected = json.load(read_file)
    if not(type_predictions == expected):
        # prettyprint new JSON, omiting optional BOM char
        with open(predictions_file + '.new', 'w', encoding='utf-8-sig') as write_file:
            json.dump(type_predictions, write_file, indent=2, sort_keys=True, ensure_ascii=False)
        raise Exception(f'{predictions_file} comparison failed.')

    evaluate_predictions(annotations, type_predictions)

    notebook_tests()


if __name__ == "__main__":
    main()
    print("Tests passed.")
