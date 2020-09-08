import clevercsv as csv
import joblib
import json
import os
import pandas as pd
from ptype.Ptype import Ptype
from tests.utils import get_datasets, evaluate_predictions


def read_data(_data_path, dataset_name):
    # wrong encoding leads to additional characters in the dataframe columns
    if dataset_name in ["mass_6.csv"]:
        encoding = "ISO-8859-1"
    else:
        encoding = "utf-8"
    return csv.csv2df(
        _data_path + dataset_name, encoding=encoding, dtype=str, skipinitialspace=True
    )


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


def get_predictions(dataset_name, infer_canonical_types=False):
    data_folder = "data/"
    model_folder = "models/"
    df = read_data(data_folder, dataset_name)
    normalizer = joblib.load(model_folder + "robust_scaler.pkl")
    clf = joblib.load(model_folder + "LR.sav")

    ptype = Ptype(
        _types={
            1: "integer",
            2: "string",
            3: "float",
            4: "boolean",
            5: "date-iso-8601",
            6: "date-eu",
            7: "date-non-std-subtype",
            8: "date-non-std",
        }
    )
    ptype.run_inference(_data_frame=df)

    if infer_canonical_types:
        for col_name in ptype.cols:
            # normalize the features as done before, then reclassify the column
            features = ptype.features[col_name]
            features[[7, 8]] = normalizer.transform(features[[7, 8]].reshape(1, -1))[0]
            ptype.cols[col_name].predicted_type = clf.predict(features.reshape(1, -1))[
                0
            ]
            # make the posterior over canonical types

    df_missing = df.apply(as_missing(ptype), axis=0)
    df_anomaly = df.apply(as_anomaly(ptype), axis=0)
    df_normal = df.apply(as_normal(ptype), axis=0)
    print(dataset_name)
    print("Original data:\n", df)
    print("Missing data:\n", df_missing)
    print("Anomalies:\n", df_anomaly)
    print("Normal:\n", df_normal)

    column_types = {
        col_name: col.predicted_type for col_name, col in ptype.cols.items()
    }
    row_types = {
        col_name: {v: str(s) for v, s in zip(col.unique_vals, col.unique_vals_status)}
        for col_name, col in ptype.cols.items()
    }

    return (column_types, row_types)


def check_predictions(type_predictions, expected_folder, dataset_name):
    expected_file = expected_folder + "/" + os.path.splitext(dataset_name)[0] + ".json"
    with open(expected_file, "r", encoding="utf-8-sig") as read_file:
        expected = json.load(read_file)

    if not (type_predictions == expected):
        # prettyprint new JSON, omiting optional BOM char
        with open(expected_file + ".new", "w", encoding="utf-8-sig") as write_file:
            json.dump(
                type_predictions,
                write_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        raise Exception(f"{expected_file} comparison failed.")


def notebook_tests():
    import os

    if os.system("pytest --nbval notebooks/*.ipynb") != 0:
        raise Exception("Notebook test(s) failed.")


def main():
    expected_folder = "tests/expected"
    annotations = json.load(open("annotations/annotations.json"))

    type_predictions = {}
    for dataset_name in get_datasets():
        col_predictions, row_predictions = get_predictions(dataset_name)

        type_predictions[dataset_name] = col_predictions
        check_predictions(col_predictions, expected_folder, dataset_name)

        check_predictions(row_predictions, expected_folder + "/row_types", dataset_name)

        col_predictions, _ = get_predictions(dataset_name, True)
        check_predictions(col_predictions, expected_folder + "/canonical", dataset_name)

    evaluate_predictions(annotations, type_predictions)

    # notebook_tests()


if __name__ == "__main__":
    main()
    print("Tests passed.")
