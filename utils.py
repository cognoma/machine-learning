def sanitize_dataframe(df):
    """Sanitize a DataFrame to prepare it for serialization.
    * Make a copy
    * Raise ValueError if it has a hierarchical index.
    * Convert categoricals to strings.
    * Convert np.int dtypes to Python int objects
    * Convert floats to objects and replace NaNs by None.
    * Convert DateTime dtypes into appropriate string representations
    
    Args:
        df (object): can be a variety of dataframe like objects
        
    Returns:
        pandas.DataFrame: a dataframe that is ready to be serialized
    """
    import pandas as pd
    import numpy as np

    df = df.copy()

    for col_name, dtype in df.dtypes.iteritems():
        if str(dtype) == 'category':
            # XXXX: work around bug in to_json for categorical types
            # https://github.com/pydata/pandas/issues/10778
            df[col_name] = df[col_name].astype(str)
        elif np.issubdtype(dtype, np.integer):
            # convert integers to objects; np.int is not JSON serializable
            df[col_name] = df[col_name].astype(object)
        elif np.issubdtype(dtype, np.floating):
            # For floats, convert nan->None: np.float is not JSON serializable
            col = df[col_name].astype(object)
            df[col_name] = col.where(col.notnull(), None)
        elif str(dtype).startswith('datetime'):
            # Convert datetimes to strings
            # astype(str) will choose the appropriate resolution
            df[col_name] = df[col_name].astype(str).replace('NaT', '')
    return df

def fill_spec_with_data(spec, data):
    """Take a Vega specification with missing data elements and complete it
    * Uses an incomplete Vega specification that has named data arguments
        which have empty values and replaces those missing values with
        data elements passed in
   * The Vega specification needs to have named data arguments that match
        the data that is passed in to this function
        
    Args:
        spec (dictionary): the Vega specification with incomplete data
        data (named dictionary of pandas.DataFrame): data elements
        
    Returns:
        dictionary: a completed Vega specification ready to be used
    """
    data_names = [d['name'] for d in spec['data']]
    
    for name, df in data.items():
        if name not in data_names:
            raise ValueError('Name not in data spec')
        for ix in range(len(data_names)):
            if data_names[ix] == name:
                break
                
        df = sanitize_dataframe(df)
        spec['data'][ix]['values'] = df.to_dict(orient='records')

    return spec
