# ptype-demo

## Initialization
```
ptype = Ptype() 
```
By default, ptype considers the following data types: 'integer', 'string', 'float', 'boolean', 'gender', 'date'. However, this can be extended using the parameter named '_types'.


```
ptype = Ptype(_types={1:'integer', 2:'string', 3:'float', 4:'boolean', 5:'gender', 6:'date-iso-8601', 7:'date-eu', 8:'date-non-std-subtype', 9:'date-non-std', 10:'IPAddress', 11:'EmailAddress') 
```

## Running
```
ptype.set_data(_data_frame=df)
ptype.run_all_columns()
```

## Summarizing the Results
We can generate a human-readable description of the predictions. For each data column, we print the posterior distribution of the column type, the most likely column type, missing or anomalies entries and the fractions of normal, missing and anomalous entries in the column.
```python
ptype.show_results()
```

Below, we present the description generated for the second column named '1':
```
col: 1
	predicted type: integer
	posterior probs:  [1.00000000e+00 0.00000000e+00 4.73609772e-47 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00]
	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 

	some normal data values:  ['101', '102', '103', '104', '106', '107', '108', '110', '113', '115', '118', '119', '121', '122', '125', '128', '129', '134', '137', '142']
	their counts:  [3, 5, 5, 6, 4, 1, 2, 2, 2, 3, 4, 2, 1, 4, 3, 6, 2, 6, 3, 1]
	percentage of normal: 0.8 

	missing values: ['?']
	their counts:  [41]
	percentage of missing: 0.2
```


Alternatively, you can find the columns that contain missing data or anomalies, and only show the results for these columns.
```python
column_names = ptype.get_columns_with_missing()
ptype.show_results(column_names)
```

    # columns with missing data: 8 
    
    col: 0
    	predicted type: integer
    	posterior probs:  [9.99999674e-01 0.00000000e+00 3.26244845e-07 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['-2', '0', '1', '2', '3']
    	their counts:  [3, 67, 54, 32, 27]
    	percentage of normal: 0.89 
    
    	missing values: ['-1']
    	their counts:  [22]
    	percentage of missing: 0.11 
    
    col: 1
    	predicted type: integer
    	posterior probs:  [1.00000000e+00 0.00000000e+00 4.73609772e-47 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['101', '102', '103', '104', '106', '107', '108', '110', '113', '115', '118', '119', '121', '122', '125', '128', '129', '134', '137', '142']
    	their counts:  [3, 5, 5, 6, 4, 1, 2, 2, 2, 3, 4, 2, 1, 4, 3, 6, 2, 6, 3, 1]
    	percentage of normal: 0.8 
    
    	missing values: ['?']
    	their counts:  [41]
    	percentage of missing: 0.2 
    
    col: 5
    	predicted type: string
    	posterior probs:  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['four', 'two']
    	their counts:  [114, 89]
    	percentage of normal: 0.99 
    
    	missing values: ['?']
    	their counts:  [2]
    	percentage of missing: 0.01 
    
    col: 18
    	predicted type: float
    	posterior probs:  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['2.54', '2.68', '2.91', '2.92', '2.97', '2.99', '3.01', '3.03', '3.05', '3.08', '3.13', '3.15', '3.17', '3.19', '3.24', '3.27', '3.31', '3.33', '3.34', '3.35']
    	their counts:  [1, 1, 7, 1, 12, 1, 5, 12, 6, 1, 2, 15, 3, 20, 2, 7, 8, 2, 1, 4]
    	percentage of normal: 0.98 
    
    	missing values: ['?']
    	their counts:  [4]
    	percentage of missing: 0.02 
    
    col: 19
    	predicted type: float
    	posterior probs:  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['2.07', '2.19', '2.36', '2.64', '2.68', '2.76', '2.80', '2.87', '2.90', '3.03', '3.07', '3.08', '3.10', '3.11', '3.12', '3.15', '3.16', '3.19', '3.21', '3.23']
    	their counts:  [1, 2, 1, 11, 2, 1, 2, 1, 3, 14, 6, 2, 2, 6, 1, 14, 1, 6, 1, 14]
    	percentage of normal: 0.98 
    
    	missing values: ['?']
    	their counts:  [4]
    	percentage of missing: 0.02 
    
    col: 21
    	predicted type: integer
    	posterior probs:  [1.00000000e+00 0.00000000e+00 3.49354091e-51 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['100', '101', '102', '106', '110', '111', '112', '114', '115', '116', '120', '121', '123', '134', '135', '140', '142', '143', '145', '152']
    	their counts:  [2, 6, 5, 1, 8, 4, 2, 6, 1, 9, 1, 3, 4, 1, 1, 1, 1, 1, 5, 3]
    	percentage of normal: 0.99 
    
    	missing values: ['?']
    	their counts:  [2]
    	percentage of missing: 0.01 
    
    col: 22
    	predicted type: integer
    	posterior probs:  [1.00000000e+00 0.00000000e+00 8.93961265e-97 0.00000000e+00
     0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
     0.00000000e+00]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['4150', '4200', '4250', '4350', '4400', '4500', '4650', '4750', '4800', '4900', '5000', '5100', '5200', '5250', '5300', '5400', '5500', '5600', '5750', '5800']
    	their counts:  [5, 5, 3, 4, 3, 7, 1, 4, 36, 1, 27, 3, 23, 7, 1, 13, 37, 1, 1, 7]
    	percentage of normal: 0.99 
    
    	missing values: ['?']
    	their counts:  [2]
    	percentage of missing: 0.01 
    
    col: 25
    	predicted type: integer
    	posterior probs:  [1.00000000e+000 0.00000000e+000 1.45025991e-107 0.00000000e+000
     0.00000000e+000 0.00000000e+000 0.00000000e+000 0.00000000e+000
     0.00000000e+000]
    	types:  ['integer', 'string', 'float', 'boolean', 'gender', 'date-iso-8601', 'date-eu', 'date-non-std-subtype', 'date-non-std'] 
    
    	some normal data values:  ['10198', '10245', '10295', '10345', '10595', '10698', '10795', '10898', '10945', '11048', '11199', '11245', '11248', '11259', '11549', '11595', '11694', '11845', '11850', '11900']
    	their counts:  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    	percentage of normal: 0.98 
    
    	missing values: ['?']
    	their counts:  [4]
    	percentage of missing: 0.02 
    	
```python
column_names = ptype.get_columns_with_anomalies()
ptype.show_results(column_names)
```

## Interactions
