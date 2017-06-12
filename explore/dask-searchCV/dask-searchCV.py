
# coding: utf-8

# # Implementing dask-searchCV to speed-up gridsearchCV and including PCA in the pipeline

# This is just a quick notebook to see:
# * How much does dask-searchCV speed things up? (A: More than 3X faster for the 2.TCGA-MLexample)
# * Can you include PCA (and a search for n_components) in the pipeline? (A: Yes!)
# * Are there any limitations to including PCA in the pipeline? (A: It takes a while & my computer couldn't handle n_jobs > 1)
# 

# ## Outline
# 1. Imports, load data, etc.
# 2. Evaluate dask-searchCV on original notebook 2 pipeline (mad feature selection)
#     * SciKit-Learn gridsearchCV (~ 7 minutes)
#     * Dask-searchCV (~ 2 minutes)
#     * Dask-searchCV with cv=10 (
# 3. Evaluate including PCA in the pipeline (note: These instances set n_jobs to 1 because, except for the trivial case, my computer froze when trying to implement with n_jobs set to -1)
#     * Trivial/Benchmark Case [2, 4]
#     * Trivial/Benchmark Case [2, 4] with cv=10
#     * Long list of few components [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
#     * Short list of many comonents [3000, 5000, 7000]
#     * Full Sweep [20, 30, 45, 67, 100, 150, 225, 337, 505, 757, 1135, 1702, 2553, 3829]

# # 1. Imports, load data, etc.

# In[1]:

get_ipython().run_cell_magic('time', '', 'import os\nimport random\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn import preprocessing\nfrom sklearn.linear_model import SGDClassifier\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.model_selection import GridSearchCV as GridSearchCV_original\nfrom sklearn.metrics import roc_auc_score, roc_curve\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.feature_selection import SelectKBest\nfrom statsmodels.robust.scale import mad\nfrom dask_searchcv import GridSearchCV as GridSearchCV_dask\nfrom sklearn.decomposition import PCA')


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[3]:

# We're going to be building a 'TP53' classifier 
GENE = '7157' # TP53


# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "try: \n    path = os.path.join('..', '..', 'download', 'expression-matrix.pkl')\n    X = pd.read_pickle(path)\nexcept:\n    path = os.path.join('..', '..', 'download', 'expression-matrix.tsv.bz2')\n    X = pd.read_table(path, index_col=0)\n\ntry:\n    path = os.path.join('..', '..', 'download', 'mutation-matrix.pkl')\n    Y = pd.read_pickle(path)\nexcept:\n    path = os.path.join('..', '..', 'download', 'mutation-matrix.tsv.bz2')\n    Y = pd.read_table(path, index_col=0)")


# In[5]:

y = Y[GENE]


# In[6]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# # 2. Evaluate dask-searchCV on original notebook 2 pipeline

# ## Median absolute deviation feature selection

# In[7]:

def fs_mad(x, y):
    """    
    Get the median absolute deviation (MAD) for each column of x
    """
    scores = mad(x) 
    return scores, np.array([np.NaN]*len(scores))


# In[8]:

# Parameter Sweep for Hyperparameters
# Modifications from orginal Notebook 2. : n_jobs set to 1 instead of -1
param_grid_original = {
    'select__k': [2000],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline_original = Pipeline(steps=[
    ('select', SelectKBest(fs_mad)),
    ('standardize', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# ## Original (SciKit-Learn)

# In[9]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_original = GridSearchCV_original(estimator=pipeline_original, param_grid=param_grid_original, n_jobs=1, scoring='roc_auc')\ncv_pipeline_original.fit(X=X_train, y=y_train)")


# ## dask-searchCV

# In[10]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_original_dask = GridSearchCV_dask(estimator=pipeline_original, param_grid=param_grid_original, n_jobs=1, scoring='roc_auc')\ncv_pipeline_original_dask.fit(X=X_train, y=y_train)")


# ## dask-searchCV with CV=10 (10 cross-validation splits as opposed to the default 3)

# In[11]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_original_dask = GridSearchCV_dask(estimator=pipeline_original, param_grid=param_grid_original, cv=10, n_jobs=1, scoring='roc_auc')\ncv_pipeline_original_dask.fit(X=X_train, y=y_train)")


# # 3. Evaluate including PCA in the pipeline

# ## Trivial/Benchmark Case [2, 4] with default cv=3

# In[12]:

param_grid = {
    'pca__n_components': [2,4],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize-pre', StandardScaler()),
    ('pca', PCA()),
    ('standardize-post', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[13]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_dask = GridSearchCV_dask(estimator=pipeline, param_grid=param_grid, n_jobs=1, scoring='roc_auc')\ncv_pipeline_dask.fit(X=X_train, y=y_train)")


# ## Trivial/Benchmark Case [2, 4] with cv=10

# In[14]:

param_grid = {
    'pca__n_components': [2,4],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize-pre', StandardScaler()),
    ('pca', PCA()),
    ('standardize-post', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[15]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_dask = GridSearchCV_dask(estimator=pipeline, param_grid=param_grid, cv=10, n_jobs=1, scoring='roc_auc')\ncv_pipeline_dask.fit(X=X_train, y=y_train)")


# ## Long list of few components [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# In[16]:

param_grid = {
    'pca__n_components': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize-pre', StandardScaler()),
    ('pca', PCA()),
    ('standardize-post', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[17]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_dask = GridSearchCV_dask(estimator=pipeline, param_grid=param_grid, n_jobs=1, scoring='roc_auc')\ncv_pipeline_dask.fit(X=X_train, y=y_train)")


# ## Short list of many comonents [3000, 5000, 7000]

# In[18]:

param_grid = {
    'pca__n_components': [3000, 5000, 7000],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize-pre', StandardScaler()),
    ('pca', PCA()),
    ('standardize-post', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[19]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_dask = GridSearchCV_dask(estimator=pipeline, param_grid=param_grid, n_jobs=1, scoring='roc_auc')\ncv_pipeline_dask.fit(X=X_train, y=y_train)")


# ## Full Sweep [20, 30, 45, 67, 100, 150, 225, 337, 505, 757, 1135, 1702, 2553, 3829]

# In[20]:

listOfComponents = []
component = 20
while component < 4000:
    listOfComponents.append(component)
    component = component + (0.5 * component)
    component = int(component)
print(listOfComponents)


# In[23]:

param_grid = {
    'pca__n_components': [20, 30, 45, 67, 100, 150, 225, 337, 505, 757, 1135, 1702, 2553, 3829],
    'classify__loss': ['log'],
    'classify__penalty': ['elasticnet'],
    'classify__alpha': [10 ** x for x in range(-3, 1)],
    'classify__l1_ratio': [0, 0.2, 0.8, 1],
}

pipeline = Pipeline(steps=[
    ('standardize-pre', StandardScaler()),
    ('pca', PCA()),
    ('standardize-post', StandardScaler()),
    ('classify', SGDClassifier(random_state=0, class_weight='balanced'))
])


# In[24]:

get_ipython().run_cell_magic('time', '', "cv_pipeline_dask = GridSearchCV_dask(estimator=pipeline, param_grid=param_grid, n_jobs=1, scoring='roc_auc')\ncv_pipeline_dask.fit(X=X_train, y=y_train)")

