import json

import requests

from cognoml.analysis import classify
from cognoml import utils

if __name__ == '__main__':
    # Create a classifier using mock input. Print output to stdout.
    url = 'https://github.com/cognoma/machine-learning/raw/876b8131bab46878cb49ae7243e459ec0acd2b47/data/api/hippo-input.json'
    response = requests.get(url)
    payload = response.json()
    payload['data_version'] = 4
    results = classify(**payload)
    json_results = json.dumps(results, indent=2, cls=utils.JSONEncoder)
    print(json_results)
