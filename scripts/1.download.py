
# coding: utf-8

# # Download cancer-data from figshare
# 
# The latest figshare data is available at https://doi.org/10.6084/m9.figshare.3487685.

# In[1]:

import os
from urllib.request import urlretrieve

import requests


# In[2]:

# Specify the figshare article ID
figshare_id = 3487685


# In[3]:

# Use the figshare API to retrieve article metadata
url = "https://api.figshare.com/v2/articles/{}".format(figshare_id)
response = requests.get(url)
response = response.json()


# In[4]:

# Show the version specific DOI
response['doi']


# In[5]:

# Make the download directory if it does not exist
if not os.path.exists('download'):
    os.mkdir('download')


# In[6]:

# Download the files specified by the metadata
for file_info in response['files']:
    url = file_info['download_url']
    name = file_info['name']
    print('Downloading {} to `{}`'.format(url, name))
    path = os.path.join('download', name)
    urlretrieve(url, path)

