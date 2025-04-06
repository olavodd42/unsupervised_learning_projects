## Projects Overview

### 1. Handwriting Recognition Using K-Means

A machine learning project that uses K-means clustering to recognize handwritten digits. This project demonstrates how unsupervised learning can be applied to image recognition tasks without relying on labeled data.

**Key Features:**
- Implementation of K-means clustering on the sklearn digits dataset
- Visual representation of cluster centers (digit prototypes)
- Comparison of different K-means initialization strategies
- Dimensionality reduction with PCA for visualization
- Application of the model to predict new handwritten digits

**Files:**
- `script.py` - Main Python script implementing K-means clustering
- `test.html` - Web interface for drawing digits to test the model

### 2. PCA Project (Telescope Data Analysis)

A data analysis project that utilizes Principal Component Analysis (PCA) to reduce dimensionality and extract important features from astronomical telescope data.

**Key Features:**
- Data standardization and preprocessing
- Full PCA implementation with variance analysis
- 2D visualization of principal components
- Comparison of classification performance using PCA vs. original features
- Implementation of Linear SVC for classification tasks

**Files:**
- `script1.py` - Initial data exploration and analysis
- script2.py - PCA implementation and classifier comparison
- data_matrix.csv - Feature data extracted from telescope observations
- classes.csv - Classification labels
- telescope_data.csv - Combined dataset with features and classes

## Running the Projects

### Requirements
```
numpy
pandas
matplotlib
sklearn
seaborn
```

### Handwriting Recognition Using K-Means
```bash
cd Handwriting_Recognition_Using_KMeans
python script.py
```
To test with custom handwritten digits, open `test.html` in a web browser.

### PCA Project
```bash
cd PCA_Project
python script2.py
```

## Learning Outcomes

These projects demonstrate:

1. **Clustering techniques** - Understanding how K-means can identify patterns in unlabeled data
2. **Dimensionality reduction** - Using PCA to extract important features and visualize high-dimensional data
3. **Model evaluation** - Comparing different initialization strategies and measuring clustering quality
4. **Data visualization** - Creating meaningful visual representations of complex datasets
5. **Feature engineering** - Understanding how transformed features can improve classifier performance

## Results

The handwriting recognition project achieves digit recognition without supervised learning by using cluster analysis to identify digit prototypes. The PCA project demonstrates significant improvement in classification performance by transforming the original feature space into principal components that better capture variance in the data.