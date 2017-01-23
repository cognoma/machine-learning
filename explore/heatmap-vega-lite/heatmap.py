
# coding: utf-8

# # This is an exercise in using Vega-Lite and Altair API to visualize some of the obtained results. 
# 
# ### Preface 
# In order to take advantage of the Altair API, the corresponding library has to be added to the environment "cognoma-machine-learning". To that end, the file environment.yml in the root directory of the "machine-learning" repo has to be modified:
# ```
# - pip:
#   ...
#   - altair==1.2.0
# ```
# After that, the environment has to be recreated, by running two commands in the shell terminal, __from the root directory of the "machine-learning" repo__:
# ```
# conda remove --name cognoma-machine-learning --all
# conda env create -f environment.yml
# ```

# ### Part 1: Recreate the previous result with Seaborn
# Now we are ready to proceed. As a starting point, we make an attempt to recreate the heatmap that shows the connection between different types of cancer and various gene mutations. This has been originally done in "3.TCGA-MLexample_Pathway.ipynb" so the first part of this notebook simply replicates those steps in order to create the original heatmap using Seaborn.

# In[2]:

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
from neo4j.v1 import GraphDatabase
get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# #### Specify model configuration - Generate genelist

# In[3]:

names = ('label', 'rel_type', 'node_id')
query_params = [
    ('Pathway', 'PARTICIPATES_GpPW', 'PC7_7459'),             # "Signaling by Hippo" - Reactome
    ('BiologicalProcess', 'PARTICIPATES_GpBP', 'GO:0035329'), # "hippo signaling" - Gene Ontology
    ('BiologicalProcess', 'PARTICIPATES_GpBP', 'GO:0035330')  # "regulation of hippo signaling" - Gene Ontology
]

param_list = [dict(zip(names, qp)) for qp in query_params]


# In[4]:

query = '''
MATCH (node)-[rel]-(gene)
WHERE node.identifier = {node_id}
  AND {label} in labels(node)
  AND {rel_type} = type(rel)
RETURN
  gene.identifier as entrez_gene_id,
  gene.name as gene_symbol
ORDER BY gene_symbol
'''


# In[5]:

driver = GraphDatabase.driver("bolt://neo4j.het.io")
full_results_df = pd.DataFrame()
with driver.session() as session:
    for parameters in param_list:
        result = session.run(query, parameters)
        result_df = pd.DataFrame((x.values() for x in result), columns=result.keys())
        full_results_df = full_results_df.append(result_df, ignore_index=True)

classifier_genes_df = full_results_df.drop_duplicates().sort_values('gene_symbol').reset_index(drop=True)
classifier_genes_df['entrez_gene_id'] = classifier_genes_df['entrez_gene_id'].astype('str')


# In[6]:

# Here are the genes that participate in the Hippo signaling pathway
classifier_genes_df


# #### Load Data

# In[7]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('../../download', 'expression-matrix.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[8]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('../../download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[9]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('../../download', 'samples.tsv')\nclinical = pd.read_table(path, index_col=0)")


# In[10]:

# Subset the Y matrix to only the genes to be classified
y_full = Y[classifier_genes_df['entrez_gene_id']]


# In[11]:

y_full.columns = classifier_genes_df['gene_symbol']
y_full = y_full.assign(disease = clinical['disease'])

# This matrix now stores the final y matrix for the classifier (y['indicator'])
y = y_full.assign(indicator = y_full.max(axis=1))


# In[12]:

unique_pos = y.groupby('disease').apply(lambda x: x['indicator'].sum())
heatmap_df0 = y_full.groupby('disease').sum().assign(TOTAL = unique_pos)
heatmap_df = heatmap_df0.divide(y_full.disease.value_counts(sort=False).sort_index(), axis=0)


# In[13]:

# What is the percentage of different mutations across different cancer types?
sns.heatmap(heatmap_df);


# ### Part 2: Building a Heatmap using Vega-Lite/Altair API
# 
# In order to take advantage of the Altair API, we have to covert the data into the so called long format (often referred to as the tidy format).

# In[14]:

# Stack it: 
heatmap_df_stacked = heatmap_df.stack()
heatmap_df_stacked.head()


# In[15]:

# The problem is that stack() produces an object 
# that is NOT Pandas Data Frame:
type(heatmap_df_stacked)


# In[16]:

# So let's convert it into the Pandas dataframe
heatmap_df_tidy = pd.DataFrame(heatmap_df_stacked)
heatmap_df_tidy.head(4)


# In[17]:

# Fix the index: Get rid of Multilevel
heatmap_df_tidy = heatmap_df_tidy.reset_index(level=['disease', 'gene_symbol'])
heatmap_df_tidy.head()


# In[19]:

# Give the third column a meaningful name: 'frequency'
heatmap_df_tidy.columns = ['disease', 'gene_symbol', 'frequency']
heatmap_df_tidy.head()


# Let's export the Pandas DataFrame into a Vega Lite JSON and save it in a file:

# In[49]:

from altair import Row, Column, Chart, Text, Data
import json
json_dump_kwargs = {
    'ensure_ascii': False,
    'indent': 2,
    'sort_keys': True,
}
def df_to_vega_lite(df, path=None):
    """
    Export a pandas.DataFrame to a vega-lite data JSON.
    
    Params
    ------
    df : pandas.DataFrame
        dataframe to convert to JSON
    path : None or str
        if None, return the JSON str. Else write JSON to the file specified by
        path.
    Source: https://github.com/dhimmel/biorxiv-licenses/
    """
    chart = Chart(data=df)
    data = chart.to_dict()['data']['values']
    if path is None:
        return json.dumps(data, **json_dump_kwargs)
    with open(path, 'w') as write_file:
        json.dump(data, write_file, **json_dump_kwargs)
        
df_to_vega_lite(heatmap_df_tidy, path='./heatmap_data_Altair_compatible.json')


# ###  Now we are ready to build the new heatmap

# In[60]:

def heatmap(data, row, column, color, cellsize=(30, 15)):
    """Create an Altair Heat-Map

    Parameters
    ----------
    row, column, color : str
        Altair trait shorthands
    cellsize : tuple
        specify (width, height) of cells in pixels
    """
    chart = Chart(data).mark_text(
               applyColorToBackground=True,
           ).encode(
               row=row,
               column=column,
               text=Text(value=' '),
               color=color
           ).configure_scale(
               textBandWidth=cellsize[0],
               bandSize=cellsize[1]
           )
#     chart = chart.encode(column=Column(axis=Axis(labelAngle='-90', offset='25')))
    return chart
heat = heatmap(Data(url='./heatmap_data_Altair_compatible.json'), 
               column='gene_symbol', row='disease', color='frequency')
heat


# There are clearly some issues with the figure format, which proved difficult to fix at the first attempt. In order to address this problem, the changes have been made directly in the JSON file. At this intial stage, it appears reasonable to edit the JSON data in Vega-Lite editor, due to a limited functionality of Altair API (as well as the author's lack of experience with Altair). 
# 
# The following is the edited version JSON file produced in the previous cell. Since the main focus of this exercise is the figure formating, the actual data has been removed (See "values": [.......]).
# * To improve the appearance of the column labels, an attribute "axis" has been added, with two sub-attributes: "labelAngle" (rotates the labels), and "offset" (shifts the labels vertically).
# * More exploring needs to be done...

# In[ ]:

{
  "mark": "text",
  "encoding": {
    "column": {
      "field": "gene_symbol",
      "type": "nominal",
      "axis": {"labelAngle": -90,
        "offset": 25
      }
    },
    "text": {"value": " "},
    "row": {"field": "disease","type": "nominal"},
    "color": {"field": "count","type": "quantitative"}
  },
  "config": {
    "mark": {"applyColorToBackground": true},
    "cell": {"width": 500,"height": 350},
    "scale": {"textBandWidth": 30,"bandSize": 15}
  },
  "data": {
    "values": [.......]
  }
}


# In[ ]:



