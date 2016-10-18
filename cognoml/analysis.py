import collections
import os
import warnings

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from cognoml import utils
from cognoml.classifiers.logistic_regression import grid_search
from cognoml.figshare import download_files

# expression_path = os.path.join('download', 'mutation-matrix.tsv.bz2')
data_directory = "download"

def read_data(version=None):
    """
    Read data.
    """
    v_dir = download_files(directory=data_directory, article_id=3487685, version=version)
    # Read expression data
    path = os.path.join(v_dir, 'expression-matrix.tsv.bz2')
    X = pd.read_table(path, index_col=0)
    return X

def classify(sample_id, mutation_status, data_version, json_sanitize=False, **kwargs):
    """
    Perform an analysis.
    
    Parameters
    ----------
    sample_id : list
        Sample IDs of the observations.
    mutation_status : list
        Mutation status (0 or 1) of each sample.
    data_version : int
        Integer with the figshare data version.
    json_sanitize : bool
        Whether to make results JSON-serializable. If `True` DataFrames are
        converted to DataTables format.

    Returns
    -------
    results : dict
        An object of results. See `data/api/hippo-output-schema.json`
        for JSON schema.
    """
    results = collections.OrderedDict()
    
    obs_df = pd.DataFrame.from_items([
        ('sample_id', sample_id),
        ('status', mutation_status),
    ])
    
    X_whole = read_data(version=data_version)
    X = X_whole.loc[obs_df.sample_id, :]
    y = obs_df.status
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0, stratify=y)
    obs_df['testing'] = obs_df.sample_id.isin(X_test.index).astype(int)
    
    grid_search.fit(X=X_train, y=y_train)
    
    predict_df = pd.DataFrame.from_items([
        ('sample_id', X_whole.index),
        ('predicted_status', grid_search.predict(X_whole)),
    ])
    if hasattr(grid_search, 'decision_function'):
        predict_df['predicted_score'] = grid_search.decision_function(X_whole)
    if hasattr(grid_search, 'predict_proba'):
        predict_df['predicted_prob'] = grid_search.predict_proba(X_whole)[:, 1]
    
    # obs_df switches to containing non-selected samples
    obs_df = obs_df.merge(predict_df, how='right', sort=True)
    obs_df['selected'] = obs_df.sample_id.isin(sample_id).astype(int)
    for column in 'status', 'testing', 'selected':
        obs_df[column] = obs_df[column].fillna(-1).astype(int)
    obs_train_df = obs_df.query("testing == 0")
    obs_test_df = obs_df.query("testing == 1")

    #y_pred_train = obs_df.query("testing == 0").predicted_score
    #y_pred_test = obs_df.query("testing == 1").predicted_score

    dimensions = collections.OrderedDict()
    dimensions['observations_selected'] = sum(obs_df.selected == 1)
    dimensions['observations_unselected'] = sum(obs_df.selected == 0)
    dimensions['features'] = len(X.columns)
    dimensions['positives'] = sum(obs_df.status == 1)
    dimensions['negatives'] = sum(obs_df.status == 0)
    dimensions['positive_prevalence'] = y.mean()
    dimensions['training_observations'] = len(obs_train_df)
    dimensions['testing_observations'] = len(obs_test_df)
    results['dimensions'] = dimensions

    performance = collections.OrderedDict()
    for part, df in ('training', obs_train_df), ('testing', obs_test_df):
        y_true = df.status
        y_pred = df.predicted_status
        metrics = utils.class_metrics(y_true, y_pred)
        metrics.update(utils.threshold_metrics(y_true, y_pred))
        performance[part] = metrics
    performance['cv'] = {'auroc': grid_search.best_score_}
    results['performance'] = performance
    
    gs = collections.OrderedDict()
    gs['cv_scores'] = utils.cv_results_to_df(grid_search.cv_results_)
    results['grid_search'] = gs
    
    results['model'] = utils.model_info(grid_search.best_estimator_.steps[-1][1])

    feature_df = utils.get_feature_df(grid_search, X.columns)
    results['model']['features'] = feature_df

    results['observations'] = obs_df
    
    if json_sanitize:
        results = utils.make_json_serializable(results)
    
    return results
