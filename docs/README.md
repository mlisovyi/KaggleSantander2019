## Introduction.

The main purpose of this repository is to illustrate usage of [MLflow](https://mlflow.org/)
in context of a kaggle competition.
MLflow is an open-source package that simplifies machine learning lifecycle.
In this use case we consentrate on the tracking functionality
that allows to do bookkeeping of ML experiments.

What does that mean in practise?
In a kaggle challenge it is important to try out different hypotheses
(different ML model types, different hyper-parameters of the models,
different feature-engineering or feature-selection techniques).
The number of such tests quickly gros and it is hard to keep track of the changes manually.
In MLflow the user defines a set of parameters and performance metrics
and the tool takes care of keeping track of parameter changes.
It also provides a dashboard in browser to visualise and compare results of different tests.

## The data and setup.

We will use the data from the 
[Santander Customer Transaction Prediction (2019)](https://www.kaggle.com/c/santander-customer-transaction-prediction)
competition on kaggle.
The goal is:

> ... identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

In order to execute the code you will need to configure a python virtual environment.
There will be a `requirements.txt` file provided in the future to simplify setup.

## MLflow usage.
The goal is to demonstrate how to use MLflow tracking. 
A key detail is that we want to use MLflow functionalities from a jupyter notebook.
MLflow is designed to work with both python scripts as well as notebooks.
However, some info is not picked by MLflow, when executed in a notebook,
e.g. [git revision, see this issue on github](https://github.com/mlflow/mlflow/issues/973).

There are only a few steps to get you going:
1. Set up an _experiment_, e.g. 
    ```python
    mlflow.set_experiment('Cool_Experiment_Name')
    ```
    This defines a set of runs (a folder containing results)
  
2. Start a _run_:
    ```python
    with mlflow.start_run(source_type=SourceType.NOTEBOOK, source_version=kg.get_last_git_commit())
      ...
       do some ML magic and record parameters and metrics
       ...
     ```
    One run defines a single test.
    Note, that `source_type` is not very critical, but `source_version` argument is needed,
    as MLflow `v0.8.2` does not pick up the github revision from a notebook automatically.
    Here, `kg.get_last_git_commit()` is a function from [keggler](https://github.com/mlisovyi/Keggler)
    that basically reads off the latest git revision in the current directory.
    
3. Log parameters (`mlflow.log_param('seed_cv', seed_cv)`),
    metrics (`mlflow.log_metric('N_trees', n_trees_ave)`) and
    artifacts, e.g. out-of-fold and submission predictions (`mlflow.log_artifact(os.getcwd()+'/out/oof.csv')`)
