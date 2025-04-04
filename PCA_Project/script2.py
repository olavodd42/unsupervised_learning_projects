import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading and preparing data...")
data_matrix = pd.read_csv('./data_matrix.csv', index_col=0)
classes = pd.read_csv('./classes.csv', index_col=0)['class']

# ---------------
# Task 7: Calculate the standardized data matrix
# ---------------
print("\nTask 7: Standardizing data matrix")
# Standardize the data matrix
mean = data_matrix.mean(axis=0)
std = data_matrix.std(axis=0)
data_matrix_standardized = (data_matrix - mean) / std

print(f"Original data range: [{data_matrix.min().min():.2f}, {data_matrix.max().max():.2f}]")
print(f"Standardized data range: [{data_matrix_standardized.min().min():.2f}, {data_matrix_standardized.max().max():.2f}]")

# ---------------
# Task 8: Perform PCA by fitting and transforming the data matrix
# ---------------
print("\nTask 8: Performing PCA")
# Find the principal components
pca = PCA()

# Fit the standardized data and calculate the principal components
principal_components = pca.fit_transform(data_matrix_standardized)
print(f'Original number of features: {data_matrix.shape[1]}')
print(f'Number of principal components: {principal_components.shape[1]}')

# ---------------
# Task 9: Calculate eigenvalues from singular values and extract eigenvectors
# ---------------
print("\nTask 9: Extracting eigenvalues and eigenvectors")
# Find the eigenvalues from the singular values
singular_values = pca.singular_values_
eigenvalues = singular_values ** 2

# Eigenvectors are in the property `.components_` as row vectors.
eigenvectors = pca.components_.T

print(f'First 3 eigenvalues: {eigenvalues[:3]}')
print(f'Shape of eigenvectors: {eigenvectors.shape}')

# ---------------
# Task 10: Extract variance ratios
# ---------------
print("\nTask 10: Extracting variance ratios")
# Get the variance ratios from the `explained_variance_ratio_`
principal_axes_variance_ratios = pca.explained_variance_ratio_
principal_axes_variance_percents = principal_axes_variance_ratios * 100

# Display top variance contributors
print("Top variance contributors:")
for i in range(3):
    print(f'Principal component {i+1}: {principal_axes_variance_percents[i]:.2f}%')

# ---------------
# Task 11: Perform PCA with 2 components
# ---------------
print("\nTask 11: Performing PCA with 2 components")
# Initialize a PCA object with 2 components
pca_2 = PCA(n_components=2) 
 
# Fit the standardized data and calculate the principal components
principal_components_2d = pca_2.fit_transform(data_matrix_standardized)
 
print(f'Number of Principal Components Features: {principal_components_2d.shape[1]}')
print(f'Original number of features: {data_matrix_standardized.shape[1]}')
print(f'Variance explained by first two components: {pca_2.explained_variance_ratio_.sum()*100:.2f}%')

# ---------------
# Task 12: Plot principal components with class as hue
# ---------------
print("\nTask 12: Plotting principal components")
# Create a DataFrame for easier plotting
principal_components_df = pd.DataFrame({
    'PC1': principal_components_2d[:, 0],
    'PC2': principal_components_2d[:, 1],
    'class': classes
})

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='class', data=principal_components_df, 
                palette='viridis', s=70, alpha=0.7)

plt.title('Principal Component Analysis - 2D Visualization')
plt.xlabel(f'Principal Component 1 ({pca_2.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca_2.explained_variance_ratio_[1]*100:.2f}% variance)')
plt.legend(title='Class')
plt.grid(True, alpha=0.3)
plt.show()
plt.close('all')

# ---------------
# Task 13: Fit classifier with PCA features and generate score
# ---------------
print("\nTask 13: Classifying with PCA features")
# Encode classes numerically
y = classes.astype('category').cat.codes
 
# Use the principal components as X and split the data
X_pca = principal_components_2d
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.33, random_state=42)
 
# Create and train Linear SVC
svc_pca = LinearSVC(random_state=0, tol=1e-5)
svc_pca.fit(X_train_pca, y_train) 
 
# Generate a score for the testing data
score_pca = svc_pca.score(X_test_pca, y_test)
print(f'Accuracy score with 2 PCA features: {score_pca:.4f}')

# Show confusion matrix
y_pred_pca = svc_pca.predict(X_test_pca)
cm_pca = confusion_matrix(y_test, y_pred_pca)
print("Confusion matrix with PCA features:")
print(cm_pca)

# ---------------
# Task 14: Fit classifier with original features and generate score
# ---------------
print("\nTask 14: Classifying with original features")
# Select first two features from the original data
first_two_original_features = data_matrix_standardized.iloc[:, :2]
 
# Split the data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    first_two_original_features, y, test_size=0.33, random_state=42)
 
# Create and train Linear SVC
svc_orig = LinearSVC(random_state=0)
svc_orig.fit(X_train_orig, y_train)
 
# Generate a score for the testing data
score_orig = svc_orig.score(X_test_orig, y_test)
print(f'Accuracy score with 2 original features: {score_orig:.4f}')

# Show confusion matrix
y_pred_orig = svc_orig.predict(X_test_orig)
cm_orig = confusion_matrix(y_test, y_pred_orig)
print("Confusion matrix with original features:")
print(cm_orig)

# Compare the results
print("\nComparison:")
print(f"PCA features accuracy: {score_pca:.4f}")
print(f"Original features accuracy: {score_orig:.4f}")
print(f"Improvement with PCA: {(score_pca - score_orig) * 100:.2f}%")

# Create side by side confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - PCA Features')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Original Features')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()
plt.close('all')