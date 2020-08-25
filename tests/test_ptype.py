def read_data(_data_path, dataset_name):
    # wrong encoding leads to additional characters in the dataframe columns
    if dataset_name in ['mass_6.csv', ]:
        encoding = 'ISO-8859-1'
    else:
        encoding = 'utf-8'
    return csv.csv2df(_data_path + dataset_name, encoding=encoding, dtype=str, skipinitialspace=True)

def get_predictions(_data_path):
    dataset_names = get_datasets()

    # create ptype
    types = {1: 'integer', 2: 'string', 3: 'float', 4: 'boolean',
             5: 'date-iso-8601', 6: 'date-eu', 7: 'date-non-std-subtype',
             8: 'date-non-std'}
    ptype = Ptype(_types=types)

    # run ptype on each dataset
    type_predictions = {}
    for dataset_name in dataset_names:

        df = read_data(_data_path, dataset_name)
        ptype.run_inference(_data_frame=df)

        # store types
        type_predictions[dataset_name] = ptype.predicted_types

    return type_predictions


def main(_data_folder='data/',
         _annotations_file='annotations/annotations.json',
         _predictions_file='tests/column_type_predictions.json'):

    annotations = json.load(open(_annotations_file))
    type_predictions = get_predictions(_data_folder)

    with open(_predictions_file, 'r', encoding='utf-8-sig') as read_file:
        expected = json.load(read_file)
    if not(type_predictions == expected):
        # prettyprint new JSON, omiting optional BOM char
        with open(_predictions_file + '.new', 'w', encoding='utf-8-sig') as write_file:
            json.dump(type_predictions, write_file, indent=2, sort_keys=True, ensure_ascii=False)
        raise Exception(f'{_predictions_file} comparison failed.')

    evaluate_predictions(annotations, type_predictions)


if __name__ == "__main__":
    from ptype.utils import get_datasets, evaluate_predictions
    from ptype.Ptype import Ptype

    import json
    import clevercsv as csv

    main()
