# Introduction.

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

# The data and setup.

We will use the data from the 
[Santander Customer Transaction Prediction (2019)](https://www.kaggle.com/c/santander-customer-transaction-prediction)
competition on kaggle.
The goal is:

> ... identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

In order to execute the code you will need to configure a python virtual environment.
There is a `conda.yaml` file provided to install all relevant packages.
This file is part of [MLflow Project](https://mlflow.org/docs/latest/projects.html) definition.

# MLflow usage.

### Bookkeeping of individual experiments

The goal is to demonstrate how to use MLflow tracking. 
The [Process notebook](ttps://github.com/mlisovyi/KaggleSantander2019/Process.ipynb) contains an example.
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
      #do some ML magic and record parameters and metrics
      ...
     ```
    One run defines a single test.
    Note, that `source_type` is not very critical, but `source_version` argument is needed,
    as MLflow `v0.8.2` does not pick up the github revision from a notebook automatically.
    Here, `kg.get_last_git_commit()` is a function from [keggler](https://github.com/mlisovyi/Keggler)
    that basically reads off the latest git revision in the current directory.
    
3. Log parameters, metrics and artifacts, e.g. out-of-fold and submission predictions
    ```python
    mlflow.log_param('seed_cv', seed_cv) # store the random seed used by the model
    #... train a model ...
    mlflow.log_metric('N_trees', n_trees_ave) # store the average number of trees
    mlflow.log_artifact(os.getcwd()+'/out/oof.csv') # store saved oof predictions on the training data
    ```

As a result, for each model that you train you will keep track of input parameters and its performance.
In addition, files with target prediction are stored, that are relevant for [stacking](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/),
i.e. building a meta-model that is trained on predictions of other models.

### Access run results

An example of the stacking implementation is available in [Stacking notebook](https://github.com/mlisovyi/KaggleSantander2019/Stacking.ipynb)
A neat feature now is that we can pull out predictions from all individual models trained in hyperparameter optimisation.
For this, we can use the deficated API.

One starts by creating a client object:
```python
from  mlflow.tracking import MlflowClient
client = MlflowClient()
```
Then one can loop through relevant experiments:
```python
for exp in [x for x in client.list_experiments() if x.name in ['HP_RS_Stratified', 'HP_RS_Stratified_RandomForest']]:
```
Loop through the meta-information about _active_(i.e. not _deleted_) runs:
```python
exp_id = exp.experiment_id
for run_info in client.list_run_infos(exp_id, ViewType.ACTIVE_ONLY):
```
Retrieve the run using the run ID extracted from the meta-information:
```python
run_id = run_info.run_uuid
run = client.get_run(run_id)
```
At this stage we can loop through associated metrics and find the one with relevant key, e.g. `'AUC'`:
```python
[m.value for m in run.data.metrics if m.key=='AUC'][0] # there is a single metric value stored, so we access the -th element of returned list
```
Or read stored predictions for use in stacking:
```python
pd.read_csv('{}/oof.csv'.format(run_info.artifact_uri), header=None)[0].astype(np.float32)
```

Now all info that is relevant for stacking is in memory and we can combine predictions from different models. 
See the notebook for an example.

# That's it!

Let me know if you want to learn more about specific aspects.
