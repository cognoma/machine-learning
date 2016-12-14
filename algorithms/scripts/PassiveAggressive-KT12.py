
# coding: utf-8

# # Create a passive aggressive classifier to predict TP53 mutation from gene expression data in TCGA

# In[1]:

import os
import urllib
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from statsmodels.robust.scale import mad

from sklearn.linear_model import PassiveAggressiveClassifier


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# *Please look at the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html).*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/TP53) about TP53.*

# ## Load Data

# In[3]:

if not os.path.exists('data'):
    os.makedirs('data')


# In[4]:

url_to_path = {
    # X matrix
    'https://ndownloader.figshare.com/files/5514386':
        os.path.join('data', 'expression.tsv.bz2'),
    # Y Matrix
    'https://ndownloader.figshare.com/files/5514389':
        os.path.join('data', 'mutation-matrix.tsv.bz2'),
}

for url, path in url_to_path.items():
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('data', 'expression.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[6]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('data', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[7]:

# We're going to be building a 'TP53' classifier 
GENE = 'TP53'


# In[8]:

y = Y[GENE]


# In[9]:

# The Series now holds TP53 Mutation Status for each Sample
y.head(6)


# In[10]:

# Here are the percentage of tumors with NF1
y.value_counts(True)


# ## Set aside 10% of the data for testing

# In[11]:

# Typically, this can only be done where the number of mutations is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X.columns), len(X_train), len(X_test))


# ## Specify Model Features

# In[12]:

# Parameter Sweep for Hyperparameters
n_feature_kept = 500

param_fixed = {
    'fit_intercept': True,
    'random_state' : 0
}

# Lower 'C' means more regularization.

param_grid = {
    'loss'  : ['hinge', 'squared_hinge'],
    'C': [10**x for x in np.linspace(-8, 1, 10)]
    
}


# ## Median absolute deviation feature selection

# In[13]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))

# select the top features with the highest MAD
feature_select = SelectKBest(fs_mad, k=n_feature_kept)


# ## Define pipeline, and cross validation model fitting

# In[14]:

# clf is Classifier
clf = PassiveAggressiveClassifier(fit_intercept=param_fixed['fit_intercept'], random_state=param_fixed['random_state'], n_iter=8)

# joblib is used to cross-validate in parallel by setting `n_jobs=-1` in GridSearchCV
# Supress joblib warning. See https://github.com/scikit-learn/scikit-learn/issues/6370

warnings.filterwarnings('ignore', message='Changing the shape of non-C contiguous array')

clf_grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')

pipeline = make_pipeline(
    feature_select,  # Feature selection
    StandardScaler(),  # Feature scaling
    clf_grid)


# In[15]:

get_ipython().run_cell_magic('time', '', '# Fit the model (the computationally intensive part)\npipeline.fit(X=X_train, y=y_train)\nbest_clf = clf_grid.best_estimator_\nfeature_mask = feature_select.get_support()  # Get a boolean array indicating the selected features')


# In[16]:

clf_grid.best_params_


# In[17]:

best_clf


# ## History of best parameters/testing

# In[18]:

# Best params

# {'C': 0.0001, 'loss': 'squared_hinge'}
# {'C': 0.0001, 'loss': 'squared_hinge'}


# ## Visualize hyperparameters performance

# In[19]:

def grid_scores_to_df(grid_scores):
    """
    Convert a sklearn.grid_search.GridSearchCV.grid_scores_ attribute to 
    a tidy pandas DataFrame where each row is a hyperparameter-fold combinatination.
    """
    rows = list()
    for grid_score in grid_scores:
        for fold, score in enumerate(grid_score.cv_validation_scores):
            row = grid_score.parameters.copy()
            row['fold'] = fold
            row['score'] = score
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ## Process Mutation Matrix

# In[20]:

cv_score_df = grid_scores_to_df(clf_grid.grid_scores_)
cv_score_df.head(2)


# In[21]:

# Cross-validated performance distribution
facet_grid = sns.factorplot(x='C', y='score', col='loss',
    data=cv_score_df, kind='violin', size=4, aspect=1)
facet_grid.set_ylabels('F1 Micro Score');


# In[22]:

# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_score_df, values='score', index='loss', columns='C')
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization C')
ax.set_ylabel('Loss Function');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[23]:

y_pred_train = pipeline.decision_function(X_train)
y_pred_test = pipeline.decision_function(X_test)

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_train = get_threshold_metrics(y_train, y_pred_train)
metrics_test = get_threshold_metrics(y_test, y_pred_test)


# In[24]:

# Plot ROC
plt.figure()
for label, metrics in ('Training', metrics_train), ('Testing', metrics_test):
    roc_df = metrics['roc_df']
    plt.plot(roc_df.fpr, roc_df.tpr,
        label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predicting TP53 mutation from gene expression (ROC curves)')
plt.legend(loc='lower right');


# ## What are the classifier coefficients?

# In[25]:

coef_df = pd.DataFrame(best_clf.coef_.transpose(), index=X.columns[feature_mask], columns=['weight'])
coef_df['abs'] = coef_df['weight'].abs()
coef_df = coef_df.sort_values('abs', ascending=False)


# In[26]:

'{:.1%} zero coefficients; {:,} negative and {:,} positive coefficients'.format(
    (coef_df.weight == 0).mean(),
    (coef_df.weight < 0).sum(),
    (coef_df.weight > 0).sum()
)


# In[27]:

coef_df.head(10)


# ## Investigate the predictions

# In[28]:

predict_df = pd.DataFrame.from_items([
    ('sample_id', X.index),
    ('testing', X.index.isin(X_test.index).astype(int)),
    ('status', y),
    ('decision_function', pipeline.decision_function(X))
])


# In[29]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('decision_function', ascending=False).query("status == 0").head(10)


# In[30]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").decision_function, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").decision_function, hist=False, label='Positives')

