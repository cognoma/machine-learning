
# coding: utf-8

# # Create a logistic regression model to predict TP53 mutation from gene expression data in TCGA

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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from statsmodels.robust.scale import mad


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[3]:

# We're going to be building a 'TP53' classifier 
GENE = '7157' # TP53


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/TP53) about TP53*

# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('download', 'expression-matrix.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[6]:

y = Y[GENE]


# In[7]:

# The Series now holds TP53 Mutation Status for each Sample
y.head(6)


# In[8]:

# Here are the percentage of tumors with NF1
y.value_counts(True)


# ## Set aside 10% of the data for testing

# In[9]:

# Typically, this can only be done where the number of mutations is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X.columns), len(X_train), len(X_test))


# ## Median absolute deviation feature selection

# In[10]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))


# ## Define pipeline and Cross validation model fitting

# In[11]:

# Parameter Sweep for Hyperparameters
param_grid = {
    'select__k': [2000],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('select', SelectKBest(fs_mad)),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])

cv_pipeline = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')


# In[12]:

get_ipython().run_cell_magic('time', '', 'cv_pipeline.fit(X=X_train, y=y_train)')


# In[13]:

# Best Params
print('{:.3%}'.format(cv_pipeline.best_score_))

# Best Params
cv_pipeline.best_params_


# ## Visualize hyperparameters performance

# In[14]:

cv_result_df = pd.concat([
    pd.DataFrame(cv_pipeline.cv_results_),
    pd.DataFrame.from_records(cv_pipeline.cv_results_['params']),
], axis='columns')
cv_result_df.head(2)


# In[15]:

# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_result_df, values='mean_test_score', index='classify__l1_ratio', columns='classify__alpha')
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization strength multiplier (alpha)')
ax.set_ylabel('Elastic net mixing parameter (l1_ratio)');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[16]:

y_pred_train = cv_pipeline.decision_function(X_train)
y_pred_test = cv_pipeline.decision_function(X_test)

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_train = get_threshold_metrics(y_train, y_pred_train)
metrics_test = get_threshold_metrics(y_test, y_pred_test)


# In[17]:

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

# In[18]:

final_pipeline = cv_pipeline.best_estimator_
final_classifier = final_pipeline.named_steps['classify']


# In[19]:

select_indices = final_pipeline.named_steps['select'].transform(
    np.arange(len(X.columns)).reshape(1, -1)
).tolist()

coef_df = pd.DataFrame.from_items([
    ('feature', X.columns[select_indices]),
    ('weight', final_classifier.coef_[0]),
])

coef_df['abs'] = coef_df['weight'].abs()
coef_df = coef_df.sort_values('abs', ascending=False)


# In[20]:

'{:.1%} zero coefficients; {:,} negative and {:,} positive coefficients'.format(
    (coef_df.weight == 0).mean(),
    (coef_df.weight < 0).sum(),
    (coef_df.weight > 0).sum()
)


# In[21]:

coef_df.head(10)


# ## Investigate the predictions

# In[22]:

predict_df = pd.DataFrame.from_items([
    ('sample_id', X.index),
    ('testing', X.index.isin(X_test.index).astype(int)),
    ('status', y),
    ('decision_function', cv_pipeline.decision_function(X)),
    ('probability', cv_pipeline.predict_proba(X)[:, 1]),
])
predict_df['probability_str'] = predict_df['probability'].apply('{:.1%}'.format)


# In[23]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('decision_function', ascending=False).query("status == 0").head(10)


# In[24]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").decision_function, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").decision_function, hist=False, label='Positives')


# In[25]:

ax = sns.distplot(predict_df.query("status == 0").probability, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").probability, hist=False, label='Positives')

