def load_object(filename):
    with open(filename, 'rb') as output:
        obj = pickle.load(output)
    return obj

def get_datanames():
    dataset_names = []
    for file in glob.glob("data/*.csv"):
        dataset_names.append(file.split('/')[-1])

    return dataset_names

def get_predictions(_data_path, dataset_names):
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

def evaluate_predictions(_data_path, annotations, type_predictions):            
    ### the column type counts of the datasets
    [total_test, dataset_counts, total_cols] = get_type_counts(type_predictions, annotations)    
    save_df_to_csv(pd.DataFrame(dataset_counts, columns=dataset_counts.keys()), 'tests/type_distributions.csv')
    
    Js, overall_accuracy = get_evaluations(annotations, type_predictions)        
    overall_accuracy_to_print = {method: "{:.2f}".format(overall_accuracy[method] / (total_cols)) for method in overall_accuracy}
    print('overall accuracy: ', overall_accuracy_to_print)    
    print('Jaccard index values: ', {t:Js[t]['ptype'] for t in Js})
    
    df = pd.DataFrame.from_dict(Js, orient='index')
    df = pd.DataFrame.from_dict(overall_accuracy_to_print, orient='index').T.append(df)
    save_df_to_csv(df, 'tests/evaluations.csv')        

    
def main(_data_path='data/', 
         _annotations_path='annotations/annotations.json', 
         _predictions_path='tests/type_predictions.json'):
    
    annotations = json.load(open(_annotations_path))
    dataset_names = get_datanames()

    type_predictions = get_predictions(_data_path, dataset_names)        
    json.dump(type_predictions, open(_predictions_path, 'w'))

    evaluate_predictions(_data_path, annotations, type_predictions)
    
    
if __name__ == "__main__":    
    from argparse import ArgumentParser
    from src.Ptype import Ptype
    
    from src.utils import get_type_counts, save_df_to_csv, get_evaluations    
    
    import glob
    import json
    import clevercsv as csv
    import pandas as pd
    
    main()