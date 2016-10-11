import os
from urllib.request import urlretrieve
import json

import requests

def get_article_versions(article_id):
    """
    Get version_to_url dictionary for a figshare article.
    """
    url = 'https://api.figshare.com/v2/articles/{}/versions'.format(article_id)
    response = requests.get(url)
    version_to_url = {d['version']: d['url'] for d in response.json()}
    return version_to_url

def download_files(directory, article_id=3487685, version=None):
    """
    Download files for a specific figshare article_id and version. Creates a
    version-specific subdirectory in `directory` and downloads all files from
    the figshare article into this subdirectory.
    
    Parameters
    ----------
    directory : str
        Directory to download files to. Files are written inside a versioned
        directory (e.g. `v2`).
    article_id : int
        article_id on figshare
    version : int or None
        figshare version. `None` means latest.
    
    Returns
    -------
    str
        The version-specific DOI corresponding to the downloaded data.
    """
    version_to_url = get_article_versions(article_id)
    if version is None:
        version = max(version_to_url.keys())
    url = version_to_url[version]
    response = requests.get(url)
    article = response.json()
    version_directory = os.path.join(directory, 'v{}'.format(version))
    
    if not os.path.exists(version_directory):
        os.mkdir(version_directory)
    
    path = os.path.join(version_directory, 'info.json')
    with open(path, 'w') as write_file:
        json.dump(article, write_file, indent=2, ensure_ascii=False, sort_keys=True)
    
    # Download the files specified by the metadata
    for file_info in article['files']:
        name = file_info['name']
        path = os.path.join(version_directory, name)
        if os.path.exists(path):
            continue
        url = file_info['download_url']
        print('Downloading {} to `{}`'.format(url, name))
        urlretrieve(url, path)
    
    return version_directory
