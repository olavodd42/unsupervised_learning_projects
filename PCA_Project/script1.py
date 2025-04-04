import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------
# Task 1: Drop nan values and examine data
# ---------------
df = pd.read_csv('./telescope_data.csv', index_col=0)
print(f'Original dataset shape: {df.shape}')
df.dropna(inplace=True)
print(f'Dataset shape after removing nulls: {df.shape}')

print('\nTask 1: First 5 rows of dataset')
print(df.head())

# ---------------
# Task 2: Extract class column and numerical features
# ---------------
classes = df['class']
data_matrix = df.drop(columns='class')

print('\nTask 2: Data matrix shape and class distribution')
print(f'Data matrix shape: {data_matrix.shape}')
print(f'Class distribution:\n{classes.value_counts()}')

# ---------------
# Task 3: Create a correlation matrix with better visualization
# ---------------
correlation_matrix = data_matrix.corr()
plt.figure(figsize=(10, 8))
ax = plt.axes()
sns.heatmap(correlation_matrix, cmap='Greens', annot=True, fmt='.2f', 
            linewidths=0.5, ax=ax)
ax.set_title('Task 3: Feature Correlation Matrix')
plt.tight_layout()
plt.show()
plt.close('all')

# ---------------
# Task 4: Perform eigendecomposition with improved visualization
# ---------------
print('\nTask 4: Eigendecomposition')
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
# Sort eigenvalues and eigenvectors in descending order
indices = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]

print(f'Number of features: {data_matrix.shape[1]}')
print(f'Number of eigenvalues: {eigenvalues.size}')
print(f'Top 3 eigenvalues: {eigenvalues[:3]}')

# ---------------
# Task 5: Find information percentages for each eigenvalue
# ---------------
information_proportions = eigenvalues / eigenvalues.sum()
information_percents = information_proportions * 100

plt.figure(figsize=(10, 6))
plt.bar(range(len(information_percents)), information_percents, alpha=0.7)
plt.plot(information_percents, 'ro-', linewidth=2)
plt.title('Task 5: Scree Plot - Information Explained by Each Principal Component')
plt.xlabel('Principal Component Index')
plt.ylabel('Percent of Information Explained')
plt.grid(True, alpha=0.3)
plt.xticks(range(len(information_percents)))
plt.show()
plt.close('all')

# Print the top contributors
print('\nTop 3 principal components contribution:')
for i in range(3):
    print(f'PC{i+1}: {information_percents[i]:.2f}% of variance')

# ---------------
# Task 6: Find cumulative information percentages
# ---------------
cumulative_information_percents = np.cumsum(information_percents)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_information_percents, 'ro-', linewidth=2, markersize=8)
plt.hlines(y=95, xmin=0, xmax=len(information_percents), colors='blue', linestyles='dashed', label='95% threshold')
plt.vlines(x=3, ymin=0, ymax=100, colors='green', linestyles='dashed', 
          label='First 3 components')
plt.title('Task 6: Cumulative Information Percentages')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Percent of Variance Explained')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(range(len(information_percents)))
plt.yticks(np.arange(0, 101, 10))
plt.show()
plt.close('all')

# Find how many components we need to reach 95%
components_needed = np.where(cumulative_information_percents >= 95)[0][0] + 1
print(f'\nNumber of principal components needed to explain 95% of variance: {components_needed}')