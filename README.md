# Titanic
*Titanic* is a playground Kaggle competition. It requires building a binary classifier to predict who survived the infamous Titanic disaster.


## Project description
**This project features:**
- Data processing and feature engineering
- Testing of most popular ML classification and model selection algorithms
- Summary of the findings in plots and tables, including
    + variable distributions
    + feature-outcome dependencies
    + in- and out-of-sample statistics (accuracy, log-likelihood, ROC curve, etc.)
    + performance summary for all tested cases
- Classes and functions allowing to create a single framework for data classification
    + Data pipelines for feature pre-processing
    + Model classes with unified interface to for model fitting, prediction and performance

**The project structure:**
- Jupyter notebooks some use examples and results: the folder "run"
    + Analysis of the features and outcome variable: run/data_analysis.ipynb
    + Model fitting, prediction and performance comparison: run/run_model.ipynb
    + Summary of the tested cases: run/summary.ipynb
- Support classes and functions: the folder "modules"
    + Data pipelines: data_framework/data_pipeline.py
    + Classification models: model_framework/model.py
- Unit tests: the folder "tests"
- Training and test data sets: the folder "data"

**A use case of the classification model class for Random Forests:**

- fit a Random Forests classifier (with the number of leaf nodes chosen by cross-validation)
- output all available statistics to measure model performance
- write a Kaggle submission in data/description/submission

```python
np.random.seed(42)
version = 7
class_weight = 'balanced'
modelTreeCV = mdl.genModelCV(mdl.RandomForest, cv=5, grid={'max_leaf_nodes': (8, 16, 32, 64)})\
    (scale='none', n_estimators=512, max_features=None, max_depth=None, class_weight=class_weight)
modelTreeCV.fitPredict(*[x for x in dtp.featuresPipelineTrainTest(dataOr, dataTOr, version=version)])
modelTreeCV.printPlotSummary()
submission = modelTreeCV.fitPredictSubmission(*dtp.featuresPipelineTrainTest(datafinOr, datafinTOr, version=version)
submission.to_csv(os.path.join(submissionFolder, "submission_{}_v{}_{}.csv".format(
    modelTreeCV.name, version, class_weight)), index_label='PassengerId')
```

More use cases can be found in run/run_model.ipynb.

## Classification problem
**Features:** Name, Age, Sex, Ticket price, Ticket class, Ticket number, Number of family members aboard, Port of embankment

**Outcome variable:** Survived (binary)

**Tested algorithms:**
- Logistic Regression
    + Logistic
    + Logistic Ridge
    + Logistic Best Subset
    + Logistic GAM
    + Logistic Local
    + Logistic Bayesian
- kNN
- Decision Trees
    + CART
    + Random Forests
    + Boosted Trees
- SVM
- Model Selection and Stacking

**Very briefly on the results:**
- Random Forests, Boosted Trees, Logistic GAM, Logistic Local and SVM show comparable out-of-sample performance.
- Final performance strongly depends on certain prior choices, such as using balanced vs. unbalanced class weights.
- The final classifier is the model stacking a few base classifiers tested before, minimizing cross-validated "normalized" deviance.
- The current best score on Kaggle: 0.799.
- The main area to improve: analysis of efficiency of model selection based on CV and OOS performance.


## Installation
Currently unavailable

//Todo: make a package
