import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline(steps=[
    ('select', VarianceThreshold()),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier())
])

param_grid = {
    'classify__random_state': [0],
    'classify__class_weight': ['balanced'],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': 10.0 ** np.linspace(-3, 1, 10),
    'classify__l1_ratio': [0.15],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    n_jobs=-1,
    scoring='roc_auc'
)
