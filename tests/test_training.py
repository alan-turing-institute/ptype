def read_dataset(dataset_name):
    if dataset_name in ["auto"]:
        df = pd.read_csv(
            "data/" + dataset_name + ".csv",
            sep=",",
            dtype=str,
            encoding="ISO-8859-1",
            keep_default_na=False,
            header=None,
        )
    else:
        df = pd.read_csv(
            "data/" + dataset_name + ".csv",
            sep=",",
            dtype=str,
            encoding="ISO-8859-1",
            keep_default_na=False,
        )

    return df


def get_inputs(dataset_name, types, annotations_file="annotations/annotations.json"):
    annotations = json.load(open(annotations_file))
    df = read_dataset(dataset_name)
    labels = annotations[dataset_name + ".csv"]

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


def main(UNIFORMLY=False):
    dataset_names = ["accident2016", "auto", "data_gov_3397_1"]
    print_to_file("number of datasets used = " + str(len(dataset_names)))

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

    df_trainings, y_trainings = [], []
    for dataset_name in dataset_names:
        df_training, y_training = get_inputs(dataset_name, types)
        df_trainings.append(df_training)
        y_trainings.append(y_training)

    ptype = Ptype(_exp_num=0, _types=types)
    ptype.train_machines_multiple_dfs(
        df_trainings, labels=y_trainings, _uniformly=UNIFORMLY
    )


if __name__ == "__main__":
    import json
    import numpy as np
    import pandas as pd

    from ptype.Ptype import Ptype
    from ptype.utils import print_to_file

    main()
    print("Training tests passed.")
