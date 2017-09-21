
# coding: utf-8

# # Explore how many PCA Components to Keep and Hyperparameter Tuning 
# # (Queries with subset of the diseases)

# __Purpose__
# 
# Address issue #106 (https://github.com/cognoma/machine-learning/issues/106). Evaluate queries that don't inlcude all the samples but rather subset the samples by disease(s).
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
# 1. Some additional evaluation and... select a final setup.
# 2. _We could also try to add the covariates data into this evaluation and see how that changes things but I'm not planning on doing that at this point._

# ## Outline:
# 1. Imports, constants and load the data
# 2. Build the querry set
#     * List of genes
#     * Summary stats for each gene-disease combo
#     * Generate a set of queries
# 3. Evaluate queries with a subset of diseases and varying number of positives
#  - a. Define some helper functions
#  - b. See how the parameters are related using the current setup
#  - c. Evaluate how changing some of the parameters effects performance
#  - d. See if we can use a function to automatically select the number of components

# ## 1. Imports, constants and load the data

# In[1]:


import os
import time
import random
import math

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


# In[2]:


RANDOMSEED = 0


# In[3]:


get_ipython().run_cell_magic('time', '', "# Load the data\ntry: \n    path = os.path.join('download', 'expression-matrix.pkl')\n    X = pd.read_pickle(path)\nexcept:\n    path = os.path.join('download', 'expression-matrix.tsv.bz2')\n    X = pd.read_table(path, index_col=0)\n\ntry:\n    path = os.path.join('download', 'mutation-matrix.pkl')\n    y = pd.read_pickle(path)\nexcept:\n    path = os.path.join('download', 'mutation-matrix.tsv.bz2')\n    y = pd.read_table(path, index_col=0)\n    \npath = os.path.join('download', 'covariates.tsv')\ncovariates = pd.read_table(path,index_col=0)")


# ## 2. Build the query set

# In[4]:


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


# Create a dictionary of {gene: 
# {disease: 
# {total: #, positive: #, negative: #}, ...

# In[5]:


get_ipython().run_cell_magic('time', '', "disease_list_acronyms = [col for col in covariates.columns if col.startswith('acronym_')]\ndisease_list = [disease.strip('acronym_') for disease in disease_list_acronyms]\ngene_dict = {}\nfor gene in list_of_genes:\n    # Subset by gene.\n    y_gene = y[gene]    \n    gene_dict[gene] = dict.fromkeys(disease_list)\n    for disease in disease_list:\n        # Subset by disease.\n        disease_cols = [col for col in disease_list_acronyms if col.endswith(disease)]\n        has_disease = covariates[disease_cols].max(axis=1) > 0\n        disease_cols = covariates[has_disease]\n        y_gene_disease = y_gene[y_gene.index.isin(disease_cols.index)]\n        # Get query stats.\n        stats = {}\n        stats['total'] = y_gene_disease.shape[0]\n        stats['positive'] = y_gene_disease.sum()\n        stats['negative'] = stats['total'] - stats['positive']\n        gene_dict[gene][disease] = stats")


# In[6]:


'''Randomly iterate through gene/disease(s) combinations to create a list
of queries that we can use for analysis. Something like:
[ [gene, [disease1, disease2, etc], {total: #, negative: #, positive: #}], ]

Try to provide general coverage of the
different possibilities of class balance and total sample size while using
the following constraints:
1. All queries should have at least 20 positives.
2. Only use one gene per query (for simplicity).
'''

def generate_queries(randomseed = 0, positive_bound = 20):
    random.seed(randomseed)
    queries = []
    keys_gene = list(gene_dict.keys())
    random.shuffle(keys_gene)
    for gene in keys_gene:
        keys_disease = list(gene_dict[gene].keys())
        random.shuffle(keys_disease)
        total_total, total_positives, total_negatives = 0, 0, 0
        diseases = []
        for disease in keys_disease:
            if total_positives < positive_bound:
                total_total += gene_dict[gene][disease]['total']
                total_positives += gene_dict[gene][disease]['positive']
                diseases.append(disease)

        total_negatives = total_total - total_positives
        num_diseases = len(diseases)
        query = [gene, diseases, {'total': total_total, 'positive': total_positives, 'negative': total_negatives}]
        if query[2]['positive'] >= positive_bound:
            queries.append(query)
    return(queries)

query_list1 = generate_queries(randomseed = 0, positive_bound = 20)
query_list2 = generate_queries(randomseed = 1, positive_bound = 60)
query_list3 = generate_queries(randomseed = 2, positive_bound = 100)
query_list4 = generate_queries(randomseed = 3, positive_bound = 200)
query_list5 = generate_queries(randomseed = 4, positive_bound = 400)
query_list6 = generate_queries(randomseed = 5, positive_bound = 500)
query_list = query_list1 + query_list2 + query_list3 + query_list4 + query_list5 + query_list6
print('There are ' + str(len(query_list)) + ' queries in the list.')


# In[7]:


# Visuallize our query set to make sure it's not too skewed
number_of_diseases =[]
for query in query_list:
    number_of_diseases.append(len(query[1]))
number_of_diseases.sort()
plt.plot(number_of_diseases)
plt.ylabel('number of diseases')
plt.title('Diseases')
plt.show()

number_of_samples =[]
for query in query_list:
    number_of_samples.append(query[2]['total'])
number_of_samples.sort()
plt.plot(number_of_samples)
plt.title('Samples')
plt.ylabel('number of samples')
plt.show()

number_of_positives =[]
for query in query_list:
    number_of_positives.append(query[2]['positive'])
number_of_positives.sort()
plt.plot(number_of_positives)
plt.title('Positives')
plt.ylabel('number of positives')
plt.show()


# ## 3. Evaluate queries with different subsets of diseases and varying number of positives

# ### 3.a. Define some helper functions

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


def evaluate_classifier(X,
                        y,
                        list_of_queries,
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
    for query in list_of_queries:
        num_samples = query[2]['total']
        num_positives = query[2]['positive']
        
        # Subset by gene.
        y_query = y[query[0]]
        # Subset by diseases.
        disease_cols = [col for col in covariates.columns if col.endswith(tuple(query[1]))]
        has_disease = covariates[disease_cols].max(axis=1) > 0
        covariates_query = covariates[has_disease]
        X_query = X[X.index.isin(covariates_query.index)]
        y_query = y_query[y_query.index.isin(covariates_query.index)]
                
        # Test Train split
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X_query, y_query, stratify=y_query, test_size=test_size, random_state=RANDOMSEED)
        # PCA.
        scaler = StandardScaler()
        if query[2]['total']*(1-test_size)*(1-(1/3)) > 350:
            n_comp = 350
        else:
            n_comp = int(query[2]['total']*(1-test_size) - 1)
        pca = PCA(n_components = n_comp, random_state = RANDOMSEED)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        pca.fit(X_train_scaled)
        X_train = pca.transform(X_train_scaled)
        X_test_scaled = scaler.transform(X_test)
        X_test = pca.transform(X_test_scaled)
        
        if set_k_range:
            k_range = set_k_range
        else:
            k_range = k_function(num_samples=num_samples,
                                 num_positives=num_positives,
                                 )     
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
        metrics = {'num_samples': num_samples,
                   'num_positive': num_positives,
                   'balance': num_positives/num_samples,
                   'train_auroc': metrics_train['auroc'], 
                   'test_auroc': metrics_test['auroc'],
                   'n_components': cv_pipeline.best_params_['select__k'], 
                   'alpha': cv_pipeline.best_params_['classify__alpha'],
                   'overfit': overfit,
                   'n_comp_status': n_comp_status,
                   'alpha_status': alpha_status
                  }
        # Add the metrics to the dictonary.
        metrics_dict[query[0]+str(query[2]['total'])] = metrics
    # Change the metrics dict into a formatted pandas dataframe.
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.T
    metrics_df.sort_values(by='num_positive', ascending=True, inplace=True)
    metrics_df = metrics_df[['num_samples', 'num_positive', 'balance', 'n_components','n_comp_status', 'alpha', 'alpha_status','train_auroc', 'test_auroc', 'overfit']]
    
    return(metrics_df)


# In[10]:


def display_stats(metrics_df, metrics_df_tocompare = None, verbose = True):
    if verbose:
        display(metrics_df)
    # Summary for metrics_df    
    metrics_df.loc['mean'] = metrics_df.mean()
    metrics_df.loc['median'] = metrics_df.median()
    metrics_df_summary = metrics_df.loc[['mean', 'median']]
    metrics_df_summary = metrics_df_summary[['num_samples', 'num_positive', 'balance', 'n_components', 'alpha', 'train_auroc', 'test_auroc','overfit']]
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


# ### 3.b. See how the parameters are related using the current setup

# In[11]:


get_ipython().run_cell_magic('time', '', 'metrics_df_current_setup = evaluate_classifier(X = X,\n                                               y = y,\n                                               list_of_queries = query_list,\n                                               set_k_range = [20, 40],\n                                               k_function = None,\n                                               alpha_range = [10** x for x in range(-3, 1)],\n                                               l1_ratio = [0.15])\ndisplay_stats(metrics_df_current_setup, metrics_df_tocompare = None)')


# In[12]:


# Show how some of these metrics/parameters might be related
metrics_df_for_correlations = metrics_df_current_setup[['num_samples', 'num_positive', 'balance', 'n_components', 'alpha', 'train_auroc', 'test_auroc', 'overfit']]
plt.figure(figsize=(16,16))
plt.title('Correlations', y=1.05, size=15)
sns.heatmap(metrics_df_for_correlations.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, annot=True)
sns.plt.show()


# ### 3.c. Evaluate how changing some of the parameters effects performance
# 
#  - When a larger range of alpha and an L1_ratio of 0 were used for the queries with all diseases there was a large performance gain (~6%-8%)... will there be a similar performance gain with these queries?

# In[13]:


get_ipython().run_cell_magic('time', '', '# alpha and L1_ratio.\nmetrics_df = evaluate_classifier(X = X,\n                                 y = y,\n                                 list_of_queries = query_list,\n                                 set_k_range = [10, 20],\n                                 k_function = None,\n                                 alpha_range = [10** x for x in range(-10, 10)],\n                                 l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = True)')


# ## 4.d. Use a function to select k (number of PCA components)
# - You can't us n_components (which the select_K is realy just a proxy for in this notebook) larger than the smallest total query size * the test train split ratio * 1-(1/cv) (for example if there is a query that has 100 total samples and you do a 80%/20% split than the training set only has 80 samples in it and if you do 3 fold CV each training fold will only have 80*0.66 (54) samples, so you couldn't do PCA with anymore than 54 components).
#  - There is a risk of overfitting if you use too many components for more unballanced queries
#  - It requires less components to capture a reasonable amount of variance for queries with less samples
#  - We haven't seen much of a clear coorelation with the two hueristics above; understanding them quantitatively has been difficult

# In[14]:


def k_func(num_samples, num_positives):
    ''' Decide the number of PCA components based on a heuristic of total samples and class balance.
    '''
    # If there are more positives than negatives, set num_positives equal to the number of negatives
    num_positives = min(num_positives, num_samples - num_positives)
    # Rule of thumb based on number of positives
    k = 5 * math.sqrt(num_positives)
    # Adjust slightly for total number of samples
    k = k * ((num_samples / 7000)**(1./6.))
    k = [int(k)]
    return(k)


# In[15]:


def k_func_proposed(num_samples, num_positives):
    ''' Simpler function proposed for use.
    '''
    # If there are more positives than negatives, set num_positives equal to the number of negatives
    num_positives = min(num_positives, num_samples - num_positives)
    # Rule of thumb based on number of positives
    if num_positives > 500:
        k = 100
    elif num_positives > 250:
        k = 50
    else:
        k = 30
    k = [int(k)]
    return(k)


# What do the k_funcs look like?

# In[16]:


def plot_k_func(k_func):
    X_plot = [] # number of samples
    Y_plot = [] # number of positives
    Z_plot = [] # k (number of components in PCA)
    for query in query_list:
        X_plot.append(query[2]['total'])
        Y_plot.append(query[2]['positive'])
        Z_plot.append(k_func(query[2]['total'], query[2]['positive']))


    plt.scatter(X_plot, Z_plot)
    plt.title('Number of Components vs. Query Size')
    plt.ylabel('K')
    plt.xlabel('Total Number of Samples')
    plt.show()

    plt.scatter(Y_plot, Z_plot)
    plt.title('Number of Components vs. Number of Positives')
    plt.ylabel('K')
    plt.xlabel('Number of Positives')
    plt.show()


# In[17]:


plot_k_func(k_func)


# In[18]:


plot_k_func(k_func_proposed)


# In[19]:


get_ipython().run_cell_magic('time', '', '# k function.\nmetrics_df = evaluate_classifier(X = X,\n                                 y = y,\n                                 list_of_queries = query_list,\n                                 set_k_range = None,\n                                 k_function = k_func,\n                                 alpha_range = [10** x for x in range(-3, 1)],\n                                 l1_ratio = [0.15])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[20]:


get_ipython().run_cell_magic('time', '', '# k function and alpha and l1_ratio.\nmetrics_df = evaluate_classifier(X = X,\n                                 y = y,\n                                 list_of_queries = query_list,\n                                 set_k_range = None,\n                                 k_function = k_func,\n                                 alpha_range = [10** x for x in range(-10, 10)],\n                                 l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = False)')


# In[21]:


get_ipython().run_cell_magic('time', '', '# proposed simpler k function and alpha and l1_ratio.\nmetrics_df = evaluate_classifier(X = X,\n                                 y = y,\n                                 list_of_queries = query_list,\n                                 set_k_range = None,\n                                 k_function = k_func_proposed,\n                                 alpha_range = [10** x for x in range(-10, 10)],\n                                 l1_ratio = [0])\ndisplay_stats(metrics_df, metrics_df_current_setup, verbose = True)')


# In[22]:


num_pos_plot = metrics_df['num_positive'].tolist()
alpha_plot = metrics_df['alpha'].tolist()
alpha_log10_plot = [math.log(a,10) for a in alpha_plot]
plt.scatter(num_pos_plot, alpha_log10_plot)
plt.title('Number of Positives vs. alpha')
plt.ylabel('Log10 of alpha')
plt.xlabel('Number of positive samples')
plt.show()

