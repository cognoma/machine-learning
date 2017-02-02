
# coding: utf-8

# # Create a logistic regression model to predict RIT1 mutation from gene expression data in TCGA

# In[1]:

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[3]:

# We're going to be building a 'RIT1' classifier 
GENE = '6016'# RIT1


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/RIT1) about RIT1*

# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', 'download', 'expression-matrix.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', 'download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[6]:

y = Y[GENE]


# In[7]:

# The Series now holds RIT1 Mutation Status for each Sample
y.head(6)


# In[8]:

# Here is the count of tumors with RIT1
y.value_counts()


# ## Set aside 30% of the data for testing

# In[9]:

# Typically, this can only be done where the number of mutations is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X.columns), len(X_train), len(X_test))


# In[10]:

# Here is the count of tumors with RIT1 in the testing data
y_test.value_counts()


# ## Preprocessing and noise reduction

# In[11]:

pipeline = Pipeline(steps=[
    ('scale_pre', StandardScaler()),
    ('pca', PCA(n_components=50, random_state=0)),
    ('scale_post', StandardScaler()),
])

X_train_scale = pipeline.fit_transform(X_train, y_train)
X_test_scale = pipeline.transform(X_test)


# In[12]:

# Percentage of preserved variance
print('{:.4}'.format(sum(pipeline.named_steps['pca'].explained_variance_ratio_)))


# ## Parameters and Classifier Fitting

# In[13]:

param_grid = {
    'alpha': [2**x for x in range(-20, 30)],
    'l1_ratio': [0, 0.05, 0.1, 0.15]
}


# In[14]:

sss = StratifiedShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
clf = SGDClassifier(random_state=0, class_weight='balanced', loss='log', penalty='elasticnet')
cv = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc', cv=sss)


# In[15]:

get_ipython().run_cell_magic('time', '', 'cv.fit(X = X_train_scale, y=y_train)')


# In[16]:

# Best Params
print('{:.3%}'.format(cv.best_score_))

# Best Params
cv.best_params_


# ## Visualize hyperparameters performance

# In[17]:

cv_result_df = pd.concat([
    pd.DataFrame(cv.cv_results_),
    pd.DataFrame.from_records(cv.cv_results_['params']),
], axis='columns')
cv_result_df.head(2)


# In[18]:

# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_result_df, values='mean_test_score', index='l1_ratio', columns='alpha')
fig, ax = plt.subplots(figsize=(15,2))

xticks = ['2^'+str(x) for x in range(-20, 30)]
keptticks = xticks[::int(len(xticks)/10)]
xticks = ['' for y in xticks]
xticks[::int(len(xticks)/10)] = keptticks

ax = sns.heatmap(cv_score_mat, annot=False, fmt='.1%', xticklabels=xticks)
ax.set_xlabel('Regularization strength multiplier (alpha)')
ax.set_ylabel('Elastic net mixing parameter (l1_ratio)');


# ## Coefficients of the Classifier

# In[19]:

best_clf = cv.best_estimator_
coef = best_clf.coef_[0]
plt.figure(figsize = (15, 5))
colors = ["red" if coef[i] < 0 else "blue" for i in range(len(coef))]
plt.bar(np.arange(len(coef)), coef, color = colors)
plt.xticks(np.arange(1, len(coef)+1), rotation=45, ha="right");


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[20]:

y_pred_train = cv.decision_function(X_train_scale)
y_pred_test = cv.decision_function(X_test_scale)

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_train = get_threshold_metrics(y_train, y_pred_train)
metrics_test = get_threshold_metrics(y_test, y_pred_test)


# In[21]:

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
plt.title('Predicting RIT1 mutation from gene expression (ROC curves)')
plt.legend(loc='lower right');


# ## Investigate the predictions

# In[22]:

X_transformed = pipeline.transform(X)


# In[23]:

predict_df = pd.DataFrame.from_items([
    ('sample_id', X.index),
    ('testing', X.index.isin(X_test.index).astype(int)),
    ('status', y),
    ('decision_function', cv.decision_function(X_transformed)),
    ('probability', cv.predict_proba(X_transformed)[:, 1]),
])
predict_df['probability_str'] = predict_df['probability'].apply('{:.1%}'.format)


# In[24]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('decision_function', ascending=False).query("status == 0").head(10)


# In[25]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").decision_function, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").decision_function, hist=False, label='Positives')


# In[26]:

ax = sns.distplot(predict_df.query("status == 0").probability, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").probability, hist=False, label='Positives')

