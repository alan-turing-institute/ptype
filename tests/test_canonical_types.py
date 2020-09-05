def read_data(_data_path, dataset_name):
    # wrong encoding leads to additional characters in the dataframe columns
    if dataset_name in [
        "mass_6.csv",
    ]:
        encoding = "ISO-8859-1"
    else:
        encoding = "utf-8"
    return csv.csv2df(
        _data_path + dataset_name, encoding=encoding, dtype=str, skipinitialspace=True
    )


def get_predictions(_data_path, _model_folder):
    dataset_names = get_datasets()

    # create ptype
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
    ptype = Ptype(_types=types)

    normalizer = joblib.load(_model_folder + "robust_scaler.pkl")
    clf = joblib.load(_model_folder + "LR.sav")

    # run ptype on each dataset
    type_predictions = {}
    for dataset_name in dataset_names:

        df = read_data(_data_path, dataset_name)
        ptype.run_inference(_data_frame=df)

        # infer canonical types
        for column in ptype.features:
            # get features
            features = ptype.features[column]

            # normalize the features as done before
            features[[7, 8]] = normalizer.transform(features[[7, 8]].reshape(1, -1))[0]

            # classify the column
            ptype.predicted_types[column] = clf.predict(features.reshape(1, -1))[0]

        # store types
        type_predictions[dataset_name] = ptype.predicted_types

    return type_predictions


def main(
    _data_folder="data/",
    _model_folder="models/",
    _annotations_file="annotations/annotations.json",
    _predictions_file="tests/column_type_predictions.json",
):

    type_predictions = get_predictions(_data_folder, _model_folder)

    # prettyprint new JSON, omiting optional BOM char
    with open(_predictions_file + ".new", "w", encoding="utf-8-sig") as write_file:
        json.dump(
            type_predictions, write_file, indent=2, sort_keys=True, ensure_ascii=False,
        )


if __name__ == "__main__":
    import clevercsv as csv
    import json

    from ptype.utils import get_datasets, evaluate_predictions
    from ptype.Ptype import Ptype

    main()
