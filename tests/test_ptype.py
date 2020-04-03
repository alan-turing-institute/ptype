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
        df = csv.csv2df(_data_path + dataset_name, encoding='ISO-8859-1', dtype=str, skipinitialspace=True)
        ptype.run_inference(_data_frame=df)

        # store types
        type_predictions[dataset_name] = ptype.predicted_types

    return type_predictions


def main(_data_path='data/',
         _annotations_path='annotations/annotations.json',
         _predictions_path='tests/column_type_predictions.json'):

    annotations = json.load(open(_annotations_path))

    type_predictions = get_predictions(_data_path)
    json.dump(type_predictions, open(_predictions_path, 'w'))

    evaluate_predictions(_data_path, annotations, type_predictions)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.utils import get_datanames, evaluate_predictions
    from src.Ptype import Ptype

    import json
    import clevercsv as csv

    import sys; print('\n'.join(sys.path))
    main()
