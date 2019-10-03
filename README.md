# ptype-demo

## Initialization
'''
ptype = Ptype() 
'''

## Running
'''
ptype.set_data(_data_frame=df, _dataset_name=dataset_name)
ptype.run_all_columns()
'''

## Summarizing the Results
'''
ptype.show_results()
'''

## Filtering the Columns based on the Results
'''
column_names = ptype.get_columns_with_missing()
ptype.show_results(column_names)
'''

'''
column_names = ptype.get_columns_with_anomalies()
ptype.show_results(column_names)
'''

## Interactions
