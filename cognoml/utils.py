import collections
import json

import numpy as np
import pandas as pd
import sklearn

def cv_results_to_df(cv_results):
    """
    Convert a `sklearn.grid_search.GridSearchCV.cv_results_` attribute to a tidy
    pandas DataFrame where each row is a hyperparameter combinatination.
    """
    cv_results_df = pd.DataFrame(cv_results)
    columns = [x for x in cv_results_df.columns if x.startswith('param_')]
    columns += ['mean_train_score', 'mean_test_score', 'std_test_score']
    cv_results_df = cv_results_df[columns]
    return cv_results_df

def expand_grid(data_dict):
    """
    Create a dataframe from every combination of given values.
    """
    rows = itertools.product(*data_dict.values())
    grid_df = pd.DataFrame.from_records(rows, columns=data_dict.keys())
    return grid_df

def df_to_datatables(df, double_precision=5, indent=2):
    """
    Convert a pandas dataframe to a JSON object formatted for datatables input.
    """
    dump_str = df.to_json(orient='split', double_precision=double_precision)
    obj = json.loads(dump_str)
    del obj['index']
    obj = collections.OrderedDict(obj)
    obj.move_to_end('data')
    return obj

def json_sanitize(obj, object_pairs_hook=collections.OrderedDict):
    """
    Sanitize an object containing pandas/numpy objects so it's JSON
    serializable. Does not preserve order since `pandas.json.dumps()` does not
    respect OrderedDict objects. Hence, it's recommended to just use the builtin
    `json.dump` function with `cls=JSONEncoder`.
    """
    obj_str = pd.json.dumps(obj)
    print(obj_str)
    obj = json.loads(obj_str, object_pairs_hook=object_pairs_hook)
    return obj

class JSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder that supports numpy types by converting them to standard
    python types.
    """

    def default(self, o):
        if type(o).__module__ == 'numpy':
            return o.item()        
        return super().default(o)

def value_map(dictionary, function, *args, **kwargs):
    """
    Edits a dictionary-like object in place to apply a function to its values.
    """
    for key, value in dictionary.items():
        dictionary[key] = function(value, *args, **kwargs)
    return dictionary

def class_metrics(y_true, y_pred):
    metrics = collections.OrderedDict()
    metrics['precision'] = sklearn.metrics.precision_score(y_true, y_pred)
    metrics['recall'] = sklearn.metrics.recall_score(y_true, y_pred)
    metrics['f1'] = sklearn.metrics.f1_score(y_true, y_pred)
    metrics['accuracy'] = sklearn.metrics.accuracy_score(y_true, y_pred)
    # See https://github.com/scikit-learn/scikit-learn/pull/6752
    metrics['balanced_accuracy'] = sklearn.metrics.recall_score(
        y_true, y_pred, pos_label=None, average='macro')
    return metrics

def threshold_metrics(y_true, y_pred):
    metrics = collections.OrderedDict()
    metrics['auroc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)
    metrics['auprc'] = sklearn.metrics.average_precision_score(y_true, y_pred)
    return metrics

def model_info(estimator):
    model = collections.OrderedDict()
    model['class'] = type(estimator).__name__
    model['module'] = estimator.__module__
    model['parameters'] = sort_dict(estimator.get_params())
    return model

def get_feature_df(grid_search, features):
    """
    Return the feature names and coefficients from the final classifier of the
    best pipeline found by GridSearchCV. See https://git.io/vPWLI. This function
    assumes every selection step of the pipeline has a name starting with
    `select`.
    
    Params
    ------
    grid_search: GridSearchCV object
        A post-fit GridSearchCV object where the estimator is a Pipeline.
    features: list
        initial feature names
    
    Returns
    -------
    pandas.DataFrame
        Dataframe of feature name and coefficient values
    """
    features = np.array(features)
    pipeline = grid_search.best_estimator_
    for name, transformer in pipeline.steps:
        if name.startswith('select'):
            X_index = np.arange(len(features)).reshape(1, -1)
            indexes = transformer.transform(X_index).tolist()
            features = features[indexes]
    step_name, classifier = pipeline.steps[-1]
    coefficients, = classifier.coef_
    feature_df = pd.DataFrame.from_items([
        ('feature', features),
        ('coefficient', coefficients),
    ])
    return feature_df

def sort_dict(dictionary):
    """
    Return a dictionary as an OrderedDict sorted by keys.
    """
    items = sorted(dictionary.items())
    return collections.OrderedDict(items)