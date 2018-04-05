# Machine learning for Project Cognoma

This repository hosts machine learning code and discussion (see [Issues](https://github.com/cognoma/machine-learning/issues)) for Project Cognoma.

## *NOTE: This repository is no longer up-to-date with the web application*

The production notebook that is served to website users can be found in the [ml-workers repository](https://github.com/cognoma/ml-workers/blob/master/notebooks/2.mutation-classifier.ipynb).
This repository will be used for continued data exploration and new modeling approaches.

## Notebooks

The following notebooks implement the primary machine learning workflow for Cognoma:

+ [`1.download.ipynb`](1.download.ipynb): downloads the cancer datasets.
+ [`2.mutation-classifier.ipynb`](2.mutation-classifier.ipynb): builds a classifier for mutation in a given gene.
+ [`3.pathway-classifier.ipynb`](3.pathway-classifier.ipynb): builds a classifier for mutation in any gene for a given pathway.

If you've modified a notebook and are submitting a pull request, then export the notebooks to scripts:

```sh
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts *.ipynb
```

## Environment

This repository uses [conda](http://conda.pydata.org/docs/ "Conda package management system and environment management system documentation") to manage its environment and install packages.
If you don't have conda installed on your system, you can [download it here](http://conda.pydata.org/miniconda.html "Miniconda Homepage").
You can install the Python 2 or 3 version of Miniconda (or Anaconda), which determines the Python version of your root environment.
Since we create a dedicated environment for this project, named `cognoma-machine-learning` whose explicit dependencies are specified in [`environment.yml`](environment.yml), the version of your root environment will not be relevant.

With conda, you can create the `cognoma-machine-learning` environment by running the following from the root directory of this repository:

```sh
# Create or overwrite the cognoma-machine-learning conda environment
conda env create --file environment.yml
```

If `environment.yml` has changed since you created the environment, run the following update command:

```sh
conda env update --file environment.yml
```

Activate the environment by running `source activate cognoma-machine-learning` on Linux or OS X and `activate cognoma-machine-learning` on Windows.
Once this environment is active in a terminal, run `jupyter notebook` to start a notebook server.
