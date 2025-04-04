# Handwriting Recognition Using K-Means Clustering
import numpy as np
import matplotlib.pyplot as plt
from time import time
import sklearn

from sklearn import metrics, datasets
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.close('all')
# Helper function for sklearn version compatibility
def get_n_init():
    return 'auto' if sklearn.__version__ >= "1.2" else 10

# Benchmark function for K-means variants
def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [
        metrics.silhouette_score(
            data, estimator[-1].labels_, metric="euclidean", sample_size=300
        )
    ]
    
    print("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(*results))

# Function to predict and interpret handwritten digits
def predict_digits(model, samples, mapping=None):
    """
    Predicts digits using the K-means model and optional mapping
    
    Args:
        model: trained K-means model
        samples: array of handwritten digit features
        mapping: dictionary mapping cluster indices to actual digits
    
    Returns:
        Array of predicted digit values
    """
    if mapping is None:
        # Default mapping based on visually inspected cluster centers
        mapping = {0: 0, 1: 9, 2: 2, 3: 1, 4: 6, 
                  5: 8, 6: 4, 7: 5, 8: 7, 9: 3}
    
    # Get cluster assignments
    labels = model.predict(samples)
    
    # Map to actual digits
    return [mapping[label] for label in labels]

# Function to visualize digit samples
def visualize_samples(samples, predictions=None, figsize=(10, 4)):
    """Visualize digit samples with their predictions"""
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        ax.imshow(sample.reshape(8, 8), cmap=plt.cm.binary)
        if predictions:
            ax.set_title(f"Predicted: {predictions[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close('all')

# =============================================================================
# 1. LOAD AND EXPLORE THE DIGITS DATASET
# =============================================================================
print("Loading digits dataset...")
digits = datasets.load_digits()
data, labels = digits.data, digits.target

# Print dataset information
print(f"Dataset shape: {data.shape}")
print(f"Number of classes: {len(np.unique(labels))}")

# Display an Example Digit
plt.figure(figsize=(4, 4))
plt.gray()
plt.matshow(digits.images[100])
plt.title("Example Digit")
plt.show()
plt.close('all')
print(f'Label for the image: {digits.target[100]}')

# =============================================================================
# 2. K-MEANS CLUSTERING
# =============================================================================
print("\nPerforming K-means clustering...")
k = 10  # 10 clusters for 10 possible digits (0-9)
model = KMeans(n_clusters=k, n_init=get_n_init(), random_state=0)
model.fit(data)

# Visualize Cluster Centers
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
plt.suptitle('K-Means Cluster Centers (Digit Prototypes)')
for i, ax in enumerate(axes.flat):
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    ax.set_title(f"Cluster {i}")
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
plt.close('all')

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================
print("\nEvaluating different K-means initialization strategies...")
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")
print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=get_n_init(), random_state=0)
bench_k_means(kmeans, "k-means++", data, labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=get_n_init(), random_state=0)
bench_k_means(kmeans, "random", data, labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_[:n_digits], n_clusters=n_digits, n_init=1, random_state=0)
bench_k_means(kmeans, "PCA-based", data, labels)

print(82 * "_")

# =============================================================================
# 4. VISUALIZATION WITH PCA
# =============================================================================
print("\nVisualizing data with PCA reduction...")
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=get_n_init(), random_state=0)
kmeans.fit(reduced_data)

# Mesh Grid for Decision Boundary
h = 0.02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict Cluster Labels for Mesh Grid
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot Decision Boundaries
plt.figure(figsize=(10, 8))
plt.imshow(
    Z, 
    extent=(x_min, x_max, y_min, y_max), 
    cmap=plt.cm.Paired, 
    aspect="auto", 
    origin="lower"
)
plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

# Plot Centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
plt.title("K-means clustering on the digits dataset (PCA-reduced data)\nCentroids are marked with white cross")
plt.xticks(())
plt.yticks(())
plt.show()
plt.close('all')

# =============================================================================
# 5. PREDICT NEW HANDWRITTEN DIGITS
# =============================================================================
print("\nPredicting new handwritten digits...")

# New samples from test.html canvas drawings
new_samples = np.array([
[0.69,2.21,4.81,7.63,3.81,0.23,0.00,0.00,4.35,7.62,7.63,7.02,7.62,3.20,0.00,0.00,3.51,7.62,5.42,1.60,7.63,3.81,0.00,0.00,0.08,4.88,7.63,7.32,7.17,1.22,0.00,0.00,0.00,1.76,7.63,7.62,5.19,0.00,0.00,0.00,0.00,5.26,7.63,7.55,5.19,0.00,0.00,0.00,0.00,4.65,7.62,7.62,2.59,0.00,0.00,0.00,0.00,0.00,0.76,0.69,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,5.26,3.74,7.17,5.42,0.00,0.00,0.00,0.00,6.86,5.34,7.62,7.32,0.84,0.00,0.00,0.00,6.86,6.48,7.62,7.62,2.29,0.00,0.00,0.00,6.56,7.17,7.17,7.62,1.76,0.00,0.00,0.00,0.92,0.38,1.45,7.62,3.05,0.00,0.00,0.00,0.00,0.00,0.76,7.62,3.43,0.00,0.00,0.00,0.00,0.00,0.15,6.56,5.11,0.00,0.00,0.00],
[0.00,0.00,0.00,0.76,2.29,1.68,0.00,0.00,0.00,0.08,2.67,7.17,7.62,7.47,0.00,0.00,0.00,5.49,7.62,6.25,4.96,7.63,0.99,0.00,0.00,3.43,2.90,0.15,1.68,7.62,4.58,0.00,0.00,0.00,0.15,4.73,7.55,7.62,7.63,1.60,0.00,0.00,0.76,7.62,7.63,7.62,4.58,0.08,0.00,0.00,0.46,7.40,7.62,5.49,0.23,0.00,0.00,0.00,0.00,0.46,0.76,0.15,0.00,0.00],
[0.00,0.00,0.00,0.38,2.29,0.23,0.00,0.00,0.00,0.00,0.69,6.41,7.62,4.73,0.84,0.00,0.00,0.15,6.33,7.62,6.94,7.62,7.55,1.91,0.00,1.07,7.55,5.80,1.07,1.45,7.55,4.50,0.00,0.00,6.86,4.58,0.00,3.81,7.62,3.20,0.00,0.00,6.86,4.58,2.44,7.55,5.04,0.08,0.00,0.00,6.18,7.09,7.09,6.86,0.31,0.00,0.00,0.00,0.99,5.03,5.34,1.91,0.00,0.00]
])

# Reshape samples for visualization
new_samples_images = [sample.reshape(8, 8) for sample in new_samples]

# Define mapping from cluster index to actual digit based on visual inspection
cluster_to_digit = {
    0: 0, 1: 9, 2: 2, 3: 1, 4: 6, 5: 8, 6: 4, 7: 5, 8: 7, 9: 3
}

# Predict clusters for new samples
new_labels = model.predict(new_samples)

# Map predictions to actual digits
predictions = [cluster_to_digit[label] for label in new_labels]

# Display results
print("Cluster assignments:", new_labels)
print("Predicted digits:", predictions)

# Visualize input and predictions
print("\nVisualizing new handwritten digits:")
visualize_samples(new_samples, predictions)