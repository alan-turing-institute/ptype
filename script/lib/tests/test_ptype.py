def read_data(_data_path, dataset_name):
    # wrong encoding leads to additional characters in the dataframe columns
    if dataset_name in ['mass_6.csv', ]:
        encoding = 'ISO-8859-1'
    else:
        encoding = 'utf-8'
    return csv.csv2df(_data_path + dataset_name, encoding=encoding, dtype=str, skipinitialspace=True)

def get_predictions(_data_path):
    dataset_names = get_datanames()

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


def main(_data_path='data/',
         _annotations_path='annotations/annotations.json',
         _predictions_path='tests/column_type_predictions.json'):

    annotations = json.load(open(_annotations_path))

    type_predictions = get_predictions(_data_path)

    # does not write optional BOM char and perfors pretty printing for json file
    with open(_predictions_path, 'w', encoding='utf-8-sig') as write_file:
            json.dump(type_predictions, write_file, indent=2, sort_keys=True, ensure_ascii=False)

    evaluate_predictions(_data_path, annotations, type_predictions)

if __name__ == "__main__":
    from ptype.utils import get_datanames, evaluate_predictions
    from ptype.Ptype import Ptype

    import json
    import clevercsv as csv

    main()
