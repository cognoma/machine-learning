# Notebooks implementing various sklearn classifiers

This directory hosts notebooks created in response to [Issue #27](https://github.com/cognoma/machine-learning/issues/27#issuecomment-238255454 "cognoma/machine-learning: Claim an sklearn algorithm to implement and troubleshoot") where team members can claim and implement an algorithm.

The recommended approach is to copy `SGDClassifier-template.ipynb` to a file named `algorithm-username.ipynb` and modify the newly created file. For example, @brankaj who's [implementing](https://github.com/cognoma/machine-learning/issues/27#issuecomment-238132718) `sklearn.linear_model.LassoCV` would name her notebook `LassoCV-brankaj.ipynb`.

Implementations are encouraged to use grid search to find optimal algorithm hyperparameters. Finally, make sure you're using the `cognoma-machine-learning` conda environment described in the [README](https://github.com/cognoma/machine-learning/tree/5eadf0309a409b08fc17d81b36b9c4bcd7acb916#environment).

Once you've finished your notebook, you should run (from this directory):

```sh
# Restart and run all cells in your notebook
jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 algorithm-username.ipynb

# Export notebook to a .py script for diff viewing
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts algorithm-username.ipynb
```

Then you can submit a pull request.
