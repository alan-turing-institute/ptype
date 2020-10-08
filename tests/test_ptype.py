import json
import jsonpickle
import numpy as np
import os
import pandas as pd

from ptype.Ptype import Ptype
from tests.utils import evaluate_predictions


types = [
    "integer",
    "string",
    "float",
    "boolean",
    "date-iso-8601",
    "date-eu",
    "date-non-std-subtype",
    "date-non-std",
]

# Associate each dataset with file prefix, encoding and header setting for Pandas read_csv.
datasets = {
    "accident2016": ("utf-8", "infer"),
    "auto": ("utf-8", None),
    "data_gov_3397_1": ("utf-8", "infer"),
    "data_gov_10151_1": ("utf-8", "infer"),
    "inspection_outcomes": ("utf-8", "infer"),
    "mass_6": ("ISO-8859-1", "infer"),
    "survey": ("utf-8", "infer"),
}


def get_predictions(dataset_name):
    df = read_dataset(dataset_name)

    ptype = Ptype(_types=types)
    schema = ptype.schema_fit(df)

    df_normal = df.apply(schema.as_normal(), axis=0)
    print(dataset_name)
    print("Original data:\n", df)
    print("Normal:\n", df_normal)

    return (
        {col_name: col.type for col_name, col in schema.cols.items()},
        {col_name: col.arff_type for col_name, col in schema.cols.items()},
        {
            col_name: {
                "missing_values": col.get_missing_values(),
                "anomalous_values": col.get_anomalous_values(),
            }
            for col_name, col in schema.cols.items()
        },
    )


def check_predictions(type_predictions, expected_folder, dataset_name):
    expected_file = expected_folder + "/" + os.path.splitext(dataset_name)[0]
    try:
        with open(expected_file + ".json", "r", encoding="utf-8-sig") as read_file:
            expected = json.load(read_file)
    except FileNotFoundError:
        expected = {}

    # JSON doesn't support integer keys
    type_predictions = {str(k): v for k, v in type_predictions.items()}
    if not (type_predictions == expected):  # dictionary comparison
        for k in type_predictions:
            if k in expected:
                if type_predictions[k] != expected[k]:
                    print(f"Differs on {k} ({type_predictions[k]} != {expected[k]})")
            else:
                print(f"Key {k} not expected")
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
        temp = [key + 1 for key, value in enumerate(types) if value == label]
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
        col_predictions, col_arff_types, missing_anomalous = get_predictions(
            dataset_name
        )

        check_predictions(col_predictions, expected_folder, dataset_name)
        check_predictions(col_arff_types, expected_folder + "/arff", dataset_name)
        check_predictions(
            missing_anomalous, expected_folder + "/missing_anomalous", dataset_name
        )

        type_predictions[dataset_name] = col_predictions

    evaluate_predictions(annotations, type_predictions)


def notebook_tests():
    import os

    if (
        os.system(
            "pytest --nbval notebooks/*.ipynb --sanitize-with script/nbval_sanitize.cfg"
        )
        != 0
    ):
        raise Exception("Notebook test(s) failed.")


def check_expected(actual, filename):
    filename_ = filename + ".json"
    with open(filename_, "r") as file:
        expected_str = file.read()
    actual_str = jsonpickle.encode(actual, indent=2, unpicklable=False)
    if expected_str != actual_str:  # deep comparison
        with open(filename_, "w") as file:
            file.write(actual_str)
        print("Result of 'git diff':")
        stream = os.popen(f"git diff {filename_}")
        output = stream.read()
        print(output)
        print(f"{filename_} comparison failed.")
        return False
    return True


def training_tests():
    df_trainings, y_trainings = [], []
    for dataset_name in ["accident2016", "auto", "data_gov_3397_1"]:
        df_training, y_training = get_inputs(dataset_name)
        df_trainings.append(df_training)
        y_trainings.append(y_training)

    ptype = Ptype(_types=types)
    initial, final, training_error = ptype.train_model(
        df_trainings, labels=y_trainings, _uniformly=False
    )

    all_passed = True
    all_passed &= check_expected(initial, "models/training_runner_initial")
    all_passed &= check_expected(final, "models/training_runner_final")
    all_passed &= check_expected(training_error, "models/training_error")
    if not all_passed:
        raise Exception("Training tests failed.")


def main():
    np.random.seed(0)
    core_tests()
    training_tests()
    notebook_tests()


if __name__ == "__main__":
    main()
    print("Tests passed.")
