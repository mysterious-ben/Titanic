# Titanic
*Titanic* is a playground Kaggle competition. It requires building a binary classifier to predict who survived the infamous Titanic disaster.


##Project description
**This project features:**
- Feature processing and feature engineering
- Test of most popular ML classification and model selection algorithms
- Summary of the findings in plots and tables, including
    + variable distributions
    + dependencies between variables
    + summary of some statistics
    + performance comparison
- Classes and functions allowing to create a single framework for data classification

**The project structure:**
- Jupyter notebooks with the results: the folder "run"
    + Analysis of the features and the outcome variable: run/data_analysis.ipynb
    + Data processing pipeline: run/run_data_pipeline.ipynb
    + Classification model performance comparison: run/run_model.ipynb
    + Kaggle submission generation: run/run_validation.ipynb
- Support classes and functions: the folder "modules"
- Unit tests: the folder "tests"
- Training and test data sets: the folder "data"


##Classificatin problem
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

**Very briefly on the results**
Random Forests, Boosted Trees, Logistic GAM, Logistic Local and SVM show comparable out-of-sample performance.
The classifier for the Kaggle submission  is a classifier stacking Logistic Regression, Random Forests, Boosted Trees, Logistic Local and SVM, with the weights minimizing a cross-validated normalized deviance loss function.
The current score on Kaggle: 0.78
The main area to improve: feature engineering
More details: see the folder "run"


##Installation
Painful //Todo: Add installation description
