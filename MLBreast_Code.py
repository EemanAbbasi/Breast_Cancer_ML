import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

scores = cross_val_score(best_model, X, y, cv=5)
print("Cross-Validation Scores:", scores)

feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

importance_df = pd.DataFrame({
    'Feature': cancer.feature_names[sorted_indices],
    'Importance': feature_importances[sorted_indices]
})

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances for Breast Cancer Dataset')
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.savefig('pca_breast_cancer.png', dpi=300, bbox_inches='tight')
plt.show()

rfe = RFE(best_model, n_features_to_select=5)
fit = rfe.fit(X, y)
print("Selected Features (RFE):", cancer.feature_names[fit.support_])

boosting_model = AdaBoostClassifier(n_estimators=50, random_state=42)
boosting_model.fit(X_train, y_train)
boosting_score = boosting_model.score(X_test, y_test)
print(f"AdaBoost Model Accuracy: {boosting_score:.2f}")
