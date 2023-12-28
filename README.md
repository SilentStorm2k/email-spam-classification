# Email Spam Classification

**A personalized content-based email spam filter using ensemble classifiers.**

## Data Description

The emails have been pre-processed into a set of features that are useful for classifying emails to this recipient as spam or not spam. Each feature corresponds to a word and denotes the proportion of all words in the email that match the given word. Each row consists of 1 email, with the first 30 entries corresponding to the features, all in the range [0, 1]. The last entry in each row is the target class, which is either 0 (not spam) or 1 (spam).

## Algorithm Choice

I chose to use a voting-based ensemble classifier that combines the following six classifiers, known for their efficacy in binary classification tasks:

- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **AdaBoost**
- **Decision Tree**

## Hyperparameter Tuning

- **RandomizedSearchCV:** Employed RandomizedSearchCV from scikit-learn for hyperparameter tuning of each classifier.
- **Iterations and Time:** Performed 100 iterations, taking approximately 4 hours on my personal computer.
- **Hyperparameter Space Exploration:** Explored roughly 1% of the entire hyperparameter space, indicating potential for further optimization.
- **Sequential Feature Selection (SFS):** Reduced the feature space to 23 features for performance enhancement. Selected features are hardcoded in the `predictTest()` function.

## Final Parameters

**SVC:**

- Gamma: 0.001
- C: 1000

**Logistic Regression:**

- C: 0.01

**Random Forest:**

- n_estimators: 200
- max_depth: 100

**K-Nearest Neighbors:**

- n_neighbors: 15

**Decision Tree:**

- max_depth: None

**AdaBoost:**

- Default hyperparameters

## Feature Selection

- **Sequential Floating Forward Selection (SFFS):** Utilized SFFS, provided by Sebastian Raschka [1], to select the 23 most informative features.
- **Feature Views and Hardcoding:** Used the `get_metric_data()` function to view selected features and hardcoded them into `predict_test()`.
- **Algorithm Availability:** SFFS algorithm is available in the `mlxtend.feature_selection` library on Raschka's GitHub page.

## Rationale for SFFS

SFFS was chosen due to its suitability for the dataset and spam email characteristics:

- **Word Combinations in Spam Emails:** SPAM emails often exhibit specific word combinations (e.g., "earn money," "get free membership").
- **Dataset Structure:** The dataset comprises word occurrences in emails.
- **Forward Selection and Correlation Identification:** SFFS performs feature selection in a forward manner, capable of identifying correlations between features (words).
- **Model Alignment:** This approach aligns well with the spam filtering model's requirements.

## Sources

[1] Sebastian Raschka. (2020, October 29). 7.7 Stacking (L07: Ensemble Methods) [Video]. YouTube. https://www.youtube.com/watch?v=8T2emza6g80

<sup>Further details and insights can be found in my [report].</sup>

[report]: https://www.mediafire.com/file/0yrlinc6u29jqhb/annotated-CSDS_340_Case_Study_1_Report.pdf/file
