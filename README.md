![build-publish on release](https://github.com/alan-turing-institute/ptype-dmkd/workflows/build-publish/badge.svg?branch=release)
![build on develop](https://github.com/alan-turing-institute/ptype-dmkd/workflows/build/badge.svg?branch=develop)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alan-turing-institute/ptype-dmkd/release?filepath=notebooks%2Fdemo.ipynb)

# Install requirements
```
pip install -r requirements.txt
```

# Usage
## Initialization
```
ptype = Ptype()
```
By default, ptype considers the following data types: integer, string, float, boolean, gender, 'date'. However, this can be extended using the parameter named ''_types''.


```
ptype = Ptype(_types={1:'integer', 2:'string', 3:'float', 4:'boolean', 5:'gender', 6:'date-iso-8601', 7:'date-eu', 8:'date-non-std-subtype', 9:'date-non-std', 10:'IPAddress', 11:'EmailAddress')
```

## Running
```
ptype.run_inference(_data_frame=df)
```

## Summarizing the Results
We can generate a human-readable description of the predictions, such as the posterior distribution of the column types, the most likely column types, missing or anomalies entries and the fractions of normal, missing and anomalous entries in the columns.

###
```python
ptype.show_results()
```
By default, it prints the descriptions for all of the columns. Alternatively, you can find the columns that contain missing data or anomalies, and only show the results for these columns.
```python
column_names = ptype.get_columns_with_missing()
ptype.show_results(column_names)
```


```python
column_names = ptype.get_columns_with_anomalies()
ptype.show_results(column_names)
```

Another way of presenting the column type predictions is to ember them in the header:
```python
ptype.show_results_df()
```

## Interactions
In addition to printing these outputs, we can change the predictions when needed.

### a. Updating the Column Type Predictions
```python
ptype.change_column_types()
```

### b. Updating the Missing Data Annotations
```python
ptype.change_missing_data_annotations()
```

### c. Updating the Anomaly Annotations
```python
ptype.change_anomaly_annotations()
```

### d. Merging Different Encodings of Missing Data
```python
ptype.merge_missing_data()
```
