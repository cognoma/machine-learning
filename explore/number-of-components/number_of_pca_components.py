
# coding: utf-8

# # Explore how many PCA Components to Keep and Hyperparameter Tuning

# ## Takeaways after running this notebook:
# 1. The accurace gain by searching over a larger range of n_components seems to be small (~1%-2% gain in testing AUROC on average), even when the range of n_components is selected for each query based on a heuristic around class balance.
# 2. There is a similar accurace gain when only the largest value in the range is used 
# 2. There is a larger performance gain if the l1_ratio is changed from 0.15 to 0 and the range of alpha is expanded (~5%-7% gain in testing AUROC on average) there isn't much performance gain if these parameters are changed independent of each other.

# __Purpose__
# 
# As described in issue #106 (https://github.com/cognoma/machine-learning/issues/106), searching over a large range of n_components (number of PCA components) causes issues with both speed and memory and may also cause issues with overfitting. This notebook is an attempt to explore the relationship between the number of components returned by PCA, the other hyperparameters and classifier performance (AUROC). Ideally, we will be able to automatically select a range of n_components to search across based on the specifics of the query. The assumption is that for the lower number of positive samples (and/or the less total samples)... the less n-components to include (i.e. a query with only 40 positive samples would use a classifer that does GridSearchCV with n_components = [30, 40, 50] whereas a query with 1,000 positive samples would use a classifier that does GridSearchCV with n_components = [100, 200, 300] _just random numbers for purpose of illustration_). This notebook attempts to provide some basis for selecting the range of n_components.
# 
# 
# __Assumptions/Notes:__
# 
# This notebook differs from the current classifier in a number of ways including:
# 1. In this notebook, PCA is performed on the entire training set prior to cross-validation rather than performed on each individual cross-validation split. This is done for simplicity and to save time and memory.
# 2. In this notebook, the covariates data is not used (for now at least).
# 
# 
# __To Do:__
# 
# 1. Evaluate queries that only use a subset of diseases or a single disease.
# 2. Try to include the covariates data and see how that affects things.
# 3. Some additional evaluation and... select a final setup.

# ## Outline:
# 1. Imports, constants and load the data
# 2. Test Train split and perform PCA
# 3. Build the querry set
# 4. Evaluate queries with all samples (i.e. all diseases) and varying number of positives
#  - a. Define some helper functions
#  - b. See how the parameters are related using the current setup
#  - c. Evaluate how changing some of the parameters effects performance
# 5. TO DO: Evaluate queries that only use a subset of diseases or a single disease
# 

# ## 1. Imports, constants and load the data

# In[1]:


import os
import time

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from dask_searchcv import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from IPython.display import display


# In[2]:


RANDOMSEED = 0


# In[3]:


get_ipython().run_cell_magic('time', '', "# Load the data\ntry: \n    path = os.path.join('download', 'expression-matrix.pkl')\n    X = pd.read_pickle(path)\nexcept:\n    path = os.path.join('download', 'expression-matrix.tsv.bz2')\n    X = pd.read_table(path, index_col=0)\n\ntry:\n    path = os.path.join('download', 'mutation-matrix.pkl')\n    y = pd.read_pickle(path)\nexcept:\n    path = os.path.join('download', 'mutation-matrix.tsv.bz2')\n    y = pd.read_table(path, index_col=0)")


# ## 2. Test Train split and perform PCA

# In[4]:


# Test Train split
X_train, X_test, y_train_allgenes, y_test_allgenes = train_test_split(X, y, test_size=0.2, random_state=RANDOMSEED)


# In[5]:


get_ipython().run_cell_magic('time', '', 'scaler = StandardScaler()\npca = PCA(n_components = 1000, random_state = RANDOMSEED)\nscaler.fit(X_train)\nX_train_scaled = scaler.transform(X_train)\npca.fit(X_train_scaled)\nX_train = pca.transform(X_train_scaled)\n\nX_test_scaled = scaler.transform(X_test)\nX_test = pca.transform(X_test_scaled)')


# ## 3. Build the query set

# In[6]:


# List of genes to iterate over (from brankaj's notebook: 
# https://github.com/cognoma/machine-learning/blob/master/explore/Classifier_results-different_genes.ipynb)
genes_LungCancer = {
    '207': 'AKT1', 
    '238': 'ALK',  
    '673': 'BRAF', 
    '4921':'DDR2',
    '1956':'EGFR',
    '2064':'ERBB2',
    '3845':'KRAS',
    '5604':'MAP2K1',
    '4893':'NRAS',
    '5290':'PIK3CA',
    '5728':'PTEN',
    '5979':'RET',
    # '6016':'RIT1', (removed because too few positives)
    '6098':'ROS1',
}

genes_TumorSuppressors = {
    '324': 'APC',  
    '672': 'BRCA1',  
    '675': 'BRCA2',
    '1029':'CDKN2A',
    '1630':'DCC',
    '4089':'SMAD4',
    '4087':'SMAD2',
    '4221':'MEN1',
    '4763':'NF1',
    '4771':'NF2',
    '7157':'TP53', 
    '5728':'PTEN', 
    '5925':'RB1',
    '7428':'VHL',
    '7486':'WRN',
    '7490':'WT1',
}

genes_Oncogenes = {
    #'5155':'PDGFB', #growth factor (removed because too few positives)
    '5159':'PDGFRB', #growth factor 
    '3791':'KDR', #receptor tyrosine kinases
    '25':'ABL1', #Cytoplasmic tyrosine kinases
    '6714':'SRC', #Cytoplasmic tyrosine kinases
    '5894':'RAF1',#cytoplasmic serine kinases
    '3265':'HRAS',#regulatory GTPases
    '4609':'MYC',#Transcription factors
    #'2353':'FOS',#Transcription factors (removed because too few positives)
    
}

list_of_genes = (list(genes_LungCancer.keys()) + list(genes_TumorSuppressors.keys()) + 
    list(genes_Oncogenes.keys()))


# In[7]:


list_of_genes_positives = []
for gene in list_of_genes:
    y_temp = y_train_allgenes[gene]
    list_of_genes_positives.append(y_temp.value_counts(True)[1]*len(y_train_allgenes))
list_of_genes = [gene for _,gene in sorted(zip(list_of_genes_positives, list_of_genes))]


# ## 4. Evaluate queries with all samples (i.e. all diseases) and varying number of positives

# ### 4.a. Define some helper functions

# In[8]:


def variance_scorer(x, y):
    """    
    Get the variance for each column of X.
    
    Because principal components have decreasing variance
    (i.e. PC4 has less variance than PC3 which has less variance
    than PC2 etc.), we can use this function in SelectKBest to select
    only the top X number of principal components.
    
    """
    scores = [np.var(column) for column in x.T]
    return scores, np.array([np.NaN]*len(scores))


# In[9]:


def evaluate_classifier(X_train, X_test,
                        y, y_train_allgenes, y_test_allgenes,
                        list_of_genes,
                        set_k_range, k_function,
                        alpha_range, 
                        l1_ratio):
    
    ''' Run a classifier setup on a set of queries.
    
        Loop through each query; train and test the classifier using the
        hyperparameters input as parameters; populate the metrics dictionary
        with some metrics of which parameters were selected and how well
        the classifier did for that query.
    '''
    
    # A dictionary to hold the performance metrics.
    metrics_dict = {}
    
    # Loop through each query; train and test the classifer; populate the metrics dictionary.
    for gene in list_of_genes:
        
        # Train and test the classifier.
        
        y_gene = y[gene]
        y_train = y_train_allgenes[gene]
        y_test = y_test_allgenes[gene]
        num_positives = int(y_gene.value_counts(True)[1]*len(y_gene))
        if set_k_range:
            k_range = set_k_range
        else:
            k_range = k_function(num_positives)     
        # Parameter Sweep for Hyperparameters
        param_grid = {
            'select__k': k_range,
            'classify__loss': ['log'],
            'classify__penalty': ['elasticnet'],
            'classify__alpha': alpha_range,
            'classify__l1_ratio': l1_ratio,
        }
        pipeline = Pipeline(steps=[
            ('select', SelectKBest(variance_scorer)),
            ('classify', SGDClassifier(random_state=RANDOMSEED, class_weight='balanced'))
        ])
        cv_pipeline = GridSearchCV(estimator=pipeline, 
                                   param_grid=param_grid,
                                   n_jobs=1, 
                                   scoring='roc_auc')
        cv_pipeline.fit(X=X_train, y=y_train)
        y_pred_train = cv_pipeline.decision_function(X_train)
        y_pred_test = cv_pipeline.decision_function(X_test)
        # Get ROC info.
        def get_threshold_metrics(y_true, y_pred):
            roc_columns = ['fpr', 'tpr', 'threshold']
            roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
            roc_df = pd.DataFrame.from_items(roc_items)
            auroc = roc_auc_score(y_true, y_pred)
            return {'auroc': auroc, 'roc_df': roc_df}
        metrics_train = get_threshold_metrics(y_train, y_pred_train)
        metrics_test = get_threshold_metrics(y_test, y_pred_test)

        # Populate the metrics dictionary.

        # Get metrics for the classifier.
        overfit = metrics_train['auroc'] - metrics_test['auroc']
        # Understand how the parameter grid worked... any params at the edge?
        if cv_pipeline.best_params_['select__k'] == min(param_grid['select__k']):
            n_comp_status = 'min'
        elif cv_pipeline.best_params_['select__k'] == max(param_grid['select__k']):
            n_comp_status = 'max'
        else:
            n_comp_status = 'OK'
        if cv_pipeline.best_params_['classify__alpha'] == min(param_grid['classify__alpha']):
            alpha_status = 'min'
        elif cv_pipeline.best_params_['classify__alpha'] == max(param_grid['classify__alpha']):
            alpha_status = 'max'
        else:
            alpha_status = 'OK'
        metrics = {'num_positive': num_positives,
                   'train_auroc': metrics_train['auroc'], 
                   'test_auroc': metrics_test['auroc'],
                   'n_components': cv_pipeline.best_params_['select__k'], 
                   'alpha': cv_pipeline.best_params_['classify__alpha'],
                   'overfit': overfit,
                   'n_comp_status': n_comp_status,
                   'alpha_status': alpha_status
                  }
        # Add the metrics to the dictonary.
        metrics_dict[gene] = metrics
    # Change the metrics dict into a formatted pandas dataframe.
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.T
    metrics_df.sort_values(by='num_positive', ascending=True, inplace=True)
    metrics_df = metrics_df[['num_positive', 'n_components','n_comp_status', 'alpha', 'alpha_status','train_auroc', 'test_auroc', 'overfit']]
    
    return(metrics_df)


# In[10]:


def display_stats(metrics_df, metrics_df_tocompare = None, verbose = True):
    if verbose:
        display(metrics_df)
    # Summary for metrics_df    
    metrics_df.loc['mean'] = metrics_df.mean()
    metrics_df.loc['median'] = metrics_df.median()
    metrics_df_summary = metrics_df.loc[['mean', 'median']]
    metrics_df_summary = metrics_df_summary[['num_positive', 'n_components', 'alpha', 'train_auroc', 'test_auroc','overfit']]
    display(metrics_df_summary)
    if metrics_df_tocompare is not None:
        # Summary for metrics_df_tocompare
        metrics_df_tocompare.loc['mean'] = metrics_df_tocompare.mean()
        metrics_df_tocompare.loc['median'] = metrics_df_tocompare.median()
        metrics_df_to_compare_summary = metrics_df_tocompare.loc[['mean', 'median']]
        # Evaluate the improvement
        mean_testing_auroc_improvement = metrics_df_summary['test_auroc']['mean'] - metrics_df_to_compare_summary['test_auroc']['mean']
        median_testing_auroc_improvement = metrics_df_summary['test_auroc']['median'] - metrics_df_to_compare_summary['test_auroc']['median']
        mean_overfit_reduction = metrics_df_to_compare_summary['overfit']['mean'] - metrics_df_summary['overfit']['mean']
        median_overfit_reduction = metrics_df_to_compare_summary['overfit']['median'] - metrics_df_summary['overfit']['median']
        print('Mean testing Auroc improved by {:.2f}%'.format(mean_testing_auroc_improvement*100))
        print('Median testing Auroc improved by {:.2f}%'.format(median_testing_auroc_improvement*100))
        print('Mean overfitting reduced by {:.1f}%'.format(mean_overfit_reduction*100))
        print('Median overfitting reduced by {:.1f}%'.format(median_overfit_reduction*100))


# ### 4.b. See how the parameters are related using the current setup

# In[11]:


get_ipython().run_cell_magic('time', '', 'metrics_df_current_setup = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [50, 100],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df_current_setup, metrics_df_tocompare = None)')


# In[12]:


# Show how some of these metrics/parameters might be related
metrics_df_for_correlations = metrics_df_current_setup[['num_positive', 'n_components', 'alpha', 'train_auroc', 'test_auroc', 'overfit']]
plt.figure(figsize=(12,12))
plt.title('Correlations', y=1.05, size=15)
sns.heatmap(metrics_df_for_correlations.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, annot=True)
sns.plt.show()


# ### 4.c. Evaluate how changing some of the parameters effects performance
# 
# The three things I'd like to evaluate are n_components, alpha and l1_ratio
# 
# My thoughts, going from easiest to most difficult to evaluate...
#  - alpha: just make the range larger if a lot of queries are at max or min (i.e. at the edge of the gridsearch space
#  - l1_ratio: the two values that we've discussed are 0 and 0.15... just try both for different setups
#  - n_components: try lists of varying size, try functions to auto select

# #### Try all combos of:
# #### k_range = [10, 20, 40, 80, 160, 320, 640]
# #### alpha_range = [10** x for x in range(-10,10)]
# #### and l1_ratio = 0

# In[13]:


get_ipython().run_cell_magic('time', '', '# k_range\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [10, 20, 40, 80, 160, 320, 640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[14]:


get_ipython().run_cell_magic('time', '', '# k_max. Test if a range is even any better than just the largest number...\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# Interesting, but maybe not too suprising. Looks like the range as opposed to just the max helps with overfitting but actually hurts with overall accurace

# In[15]:


get_ipython().run_cell_magic('time', '', '# l1_ratio\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [50, 100],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[16]:


get_ipython().run_cell_magic('time', '', '# alpha_range\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [50, 100],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[17]:


get_ipython().run_cell_magic('time', '', '# alpha_range and l1_ratio\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [50, 100],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[18]:


get_ipython().run_cell_magic('time', '', '# alpha_range and k_range\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [10, 20, 40, 80, 160, 320, 640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[19]:


get_ipython().run_cell_magic('time', '', '# k_range and l1_ratio\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [10, 20, 40, 80, 160, 320, 640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[20]:


get_ipython().run_cell_magic('time', '', '# All three: k_range, alpha_range and l1_ratio\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [10, 20, 40, 80, 160, 320, 640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[21]:


get_ipython().run_cell_magic('time', '', '# All three: k_range, alpha_range and l1_ratio... Again this time only use the max of the k_range\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = [640],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# With the larger aplha range and l1_ratio it looks like having the range of k as opposed to just the max helps with both overall performance as well as overfitting

# ### Use a function to automatically select the parameter space to search

# In[22]:


# k_range function
def k_func(num_positives):
    if num_positives < 50:
        k_range = [6, 8, 10, 14, 20, 28]
    elif num_positives < 100:
        k_range = [10, 15, 20, 30, 50, 80]
    elif num_positives < 200:
        k_range = [15, 25, 50, 75, 100, 175]
    elif num_positives < 500:
        k_range = [60, 100, 150, 250, 400] # Tried 30, 60, 100, 150
    else:
        k_range = [100, 200, 400, 800]
    return(k_range)


# In[23]:


get_ipython().run_cell_magic('time', '', '# k_range func\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = None,\n                                               k_function = k_func,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[24]:


get_ipython().run_cell_magic('time', '', '# k_range func with larger alpha range\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = None,\n                                               k_function = k_func,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[25]:


get_ipython().run_cell_magic('time', '', '# k_range func with larger alpha range and l1_ratio = 0\nmetrics_df = evaluate_classifier(X_train = X_train,\n                                               X_test = X_test,\n                                               y = y,\n                                               y_train_allgenes = y_train_allgenes,\n                                               y_test_allgenes = y_test_allgenes,\n                                               list_of_genes = list_of_genes,\n                                               set_k_range = None,\n                                               k_function = k_func,\n                                               alpha_range = [10** x for x in range(-10, 10)],\n                                               l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup)')

