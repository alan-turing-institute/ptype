import json
import jsonpickle
import numpy as np
import os
import pandas as pd

from ptype.Ptype import Ptype
from ptype.Trainer import Trainer
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
    "housing_price": ("utf-8", "infer"),
    "inspection_outcomes": ("utf-8", "infer"),
    "mass_6": ("ISO-8859-1", "infer"),
    "survey": ("utf-8", "infer"),
}


def get_predictions(dataset_name, data_folder):
    df = read_dataset(dataset_name, data_folder)

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


def read_dataset(dataset_name, data_folder):
    filename = data_folder + dataset_name + ".csv"
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


def get_inputs(
    dataset_name, annotations_file="annotations/annotations.json", data_folder="data/"
):
    annotations = json.load(open(annotations_file))
    df = read_dataset(dataset_name, data_folder)
    labels = annotations[dataset_name]

    # discard labels other than initialized 'types'
    indices = [i for i, label in enumerate(labels) if label in types]
    df = df[df.columns[np.array(indices)]]
    labels = [labels[index] for index in indices]

    # find the integer labels for the types
    y = [types.index(label) + 1 for label in labels]

    return df, y


def core_tests():
    expected_folder = "tests/expected"
    data_folder = "data/"
    annotations = json.load(open("annotations/annotations.json"))

    type_predictions = {}
    for dataset_name in datasets:
        col_predictions, col_arff_types, missing_anomalous = get_predictions(
            dataset_name, data_folder
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
    dfs, ys = [], []
    for dataset_name in ["accident2016", "auto", "data_gov_3397_1", "data_gov_10151_1"]:
        df, y = get_inputs(dataset_name)
        dfs.append(df)
        ys.append(y)

    ptype = Ptype(_types=types)
    trainer = Trainer(ptype.machines, dfs, ys)
    initial, final, training_error = trainer.train(20, False)

    all_passed = True
    all_passed &= check_expected(initial, "models/training_runner_initial")
    all_passed &= check_expected(final, "models/training_runner_final")
    all_passed &= check_expected(training_error, "models/training_error")
    if not all_passed:
        raise Exception("Training tests failed.")


def other_test():
    df = pd.read_csv(
        "data/rodents.csv", encoding="ISO-8859-1", dtype="str", keep_default_na=False
    )
    Ptype().schema_fit(df).transform(df)


def main():
    np.random.seed(0)
    other_test()
    core_tests()
    training_tests()
    notebook_tests()


if __name__ == "__main__":
    main()
    print("Tests passed.")
