# Breast Cancer Classification with Machine Learning

This project applies various machine learning techniques to the Breast Cancer Wisconsin dataset to classify whether a tumor is malignant or benign. Techniques used include Random Forest, AdaBoost, PCA (Principal Component Analysis), and Recursive Feature Elimination (RFE).

## Project Overview

The goal of this project is to explore different methods of classification and feature selection to better understand which features in the dataset are most influential and how well the models perform.

### Dataset

The dataset used is the Breast Cancer Wisconsin dataset from the `sklearn.datasets` package. The dataset contains 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The target variable indicates whether the tumor is malignant or benign.

---

## Libraries and Dependencies

The following Python libraries are used in the project:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

Ensure these packages are installed by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Steps in the Project

### 1. Data Preprocessing

The Breast Cancer dataset is loaded using `sklearn.datasets.load_breast_cancer()`. The data is split into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. Random Forest Classifier and Hyperparameter Tuning

A Random Forest model is trained on the dataset, and hyperparameters are tuned using grid search (`GridSearchCV`).

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
```

Cross-validation scores for the best model are calculated:

```python
scores = cross_val_score(best_model, X, y, cv=5)
```

### 3. Feature Importance

The feature importances from the best Random Forest model are visualized to understand which features contribute most to the classification:

```python
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Feature Importances for Breast Cancer Dataset')
```

Here is an example plot showing the feature importances:

![Feature Importance](https://github.com/EemanAbbasi/Breast_Cancer_ML/blob/main/feature_Breast.png)

### 4. Principal Component Analysis (PCA)

PCA is performed to reduce the dimensionality of the dataset, visualizing the first two principal components:

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
```

Here is an example PCA plot:

![PCA Plot](https://github.com/EemanAbbasi/Breast_Cancer_ML/blob/main/breast_PCA.png)

### 5. Recursive Feature Elimination (RFE)

RFE is applied to select the top 5 most important features in the dataset:

```python
rfe = RFE(best_model, n_features_to_select=5)
fit = rfe.fit(X, y)
print("Selected Features (RFE):", cancer.feature_names[fit.support_])
```

### 6. AdaBoost Classifier

An AdaBoost model is also trained on the dataset and tested for accuracy:

```python
boosting_model = AdaBoostClassifier(n_estimators=50, random_state=42)
boosting_score = boosting_model.score(X_test, y_test)
```

---

## Results

- **Random Forest Cross-Validation Scores:** The model performs well across multiple folds, ensuring robustness.
- **AdaBoost Accuracy:** The AdaBoost model achieves an accuracy of around `X.XX` (replace with your score) on the test set.
- **Feature Importance:** Visualization shows which features contribute the most to classification.
- **PCA:** The dataset is visualized in 2D space using PCA, allowing for exploratory data analysis.
