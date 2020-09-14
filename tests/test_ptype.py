import json
import numpy as np
import os
import pandas as pd

from ptype.Ptype import Ptype, Column2ARFF
from tests.utils import evaluate_predictions


types = {
    1: "integer",
    2: "string",
    3: "float",
    4: "boolean",
    5: "date-iso-8601",
    6: "date-eu",
    7: "date-non-std-subtype",
    8: "date-non-std",
}

# Associate each dataset with file prefix, encoding and header setting for Pandas read_csv.
datasets = {
    "accident2016":         ("utf-8", "infer"),
    "auto":                 ("utf-8", None),
    "data_gov_3397_1":      ("utf-8", "infer"),
    "data_gov_10151_1":     ("utf-8", "infer"),
    "inspection_outcomes":  ("utf-8", "infer"),
    "mass_6":               ("ISO-8859-1", "infer"),
    "survey":               ("utf-8", "infer"),
}


def as_normal(ptype):
    return lambda series: series.map(
        lambda v: v if v in ptype.cols[series.name].get_normal_predictions() else pd.NA
    )


def as_missing(ptype):
    return lambda series: series.map(
        lambda v: v
        if v in ptype.cols[series.name].get_missing_data_predictions()
        else pd.NA
    )


def as_anomaly(ptype):
    return lambda series: series.map(
        lambda v: v if v in ptype.cols[series.name].get_anomaly_predictions() else pd.NA
    )


def get_predictions(dataset_name):
    df = read_dataset(dataset_name)

    ptype = Ptype(_types=types)
    ptype.run_inference(_data_frame=df)

    column2ARFF = Column2ARFF("models/")
    for col_name in ptype.cols:
        # normalize the features as done before, then reclassify the column
        features = ptype.features[col_name]
        ptype.cols[col_name].arff_type = column2ARFF.get_arff_type(features)

    df_missing = df.apply(as_missing(ptype), axis=0)
    df_anomaly = df.apply(as_anomaly(ptype), axis=0)
    df_normal = df.apply(as_normal(ptype), axis=0)
    print(dataset_name)
    print("Original data:\n", df)
    print("Missing data:\n", df_missing)
    print("Anomalies:\n", df_anomaly)
    print("Normal:\n", df_normal)

    col_types = {col_name: col.predicted_type for col_name, col in ptype.cols.items()}
    col_arff_types = {col_name: col.arff_type for col_name, col in ptype.cols.items()}
    row_types = {
        col_name: {v: str(s) for v, s in zip(col.unique_vals, col.unique_vals_status)}
        for col_name, col in ptype.cols.items()
    }

    return (col_types, col_arff_types, row_types)


def check_predictions(type_predictions, expected_folder, dataset_name):
    expected_file = expected_folder + "/" + os.path.splitext(dataset_name)[0]
    with open(expected_file + ".json", "r", encoding="utf-8-sig") as read_file:
        expected = json.load(read_file)

    # JSON doesn't support integer keys
    type_predictions = {str(k): v for k, v in type_predictions.items()}
    if not (type_predictions == expected):
        for k in type_predictions:
            if type_predictions[k] != expected[k]:
                print("Differs on " + k)
        # prettyprint new JSON, omitting optional BOM char
        with open(expected_file + ".new.json", "w", encoding="utf-8-sig") as write_file:
            json.dump(
                type_predictions,
                write_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        raise Exception(f"{expected_file + '.json'} comparison failed.")


def read_dataset(dataset_name):
    filename = "data/" + dataset_name + ".csv"
    if dataset_name in datasets:
        encoding, header = datasets[dataset_name]
        return pd.read_csv(
            filename,
            sep=",",
            dtype=str,
            encoding=encoding,
            keep_default_na=False,
            skipinitialspace=True,
            header=header,
        )
    else:
        raise Exception(f"{filename} not known.")


def get_inputs(dataset_name, annotations_file="annotations/annotations.json"):
    annotations = json.load(open(annotations_file))
    df = read_dataset(dataset_name)
    labels = annotations[dataset_name]

    # remove unused types
    indices = []
    for i, label in enumerate(labels):
        if label not in ["all identical", "gender"]:
            indices.append(i)
    labels = [labels[index] for index in indices]
    df = df[df.columns[np.array(indices)]]

    # find the integer labels for the types
    y = []
    for label in labels:
        temp = [key for (key, value) in types.items() if value == label]
        if len(temp) != 0:
            y.append(temp[0])
        else:
            print(label, temp)

    return df, y


def core_tests():
    expected_folder = "tests/expected"
    annotations = json.load(open("annotations/annotations.json"))

    type_predictions = {}
    for dataset_name in datasets:
        col_predictions, col_arff_types, row_predictions = get_predictions(dataset_name)

        check_predictions(col_predictions, expected_folder, dataset_name)
        check_predictions(col_arff_types, expected_folder + "/arff", dataset_name)
        check_predictions(row_predictions, expected_folder + "/row_types", dataset_name)

        type_predictions[dataset_name] = col_predictions

    evaluate_predictions(annotations, type_predictions)


def notebook_tests():
    import os

    if os.system("pytest --nbval notebooks/*.ipynb -v") != 0:
        raise Exception("Notebook test(s) failed.")


def training_tests():
    # print_to_file("number of datasets used = " + str(len(dataset_names)))

    df_trainings, y_trainings = [], []
    for dataset_name in ["accident2016", "auto", "data_gov_3397_1"]:
        df_training, y_training = get_inputs(dataset_name)
        df_trainings.append(df_training)
        y_trainings.append(y_training)

    ptype = Ptype(_exp_num=0, _types=types)
    ptype.train_machines_multiple_dfs(
        df_trainings, labels=y_trainings, _uniformly=False
    )


def main():
    core_tests()
    notebook_tests()
    training_tests()


if __name__ == "__main__":
    main()
    print("Tests passed.")
