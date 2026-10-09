# Heart Disease Clustering with K-Means

## Overview

This project explores the Heart Disease dataset using Python and unsupervised machine learning techniques. It focuses on data exploration, preprocessing, visualization, and K-Means clustering to identify groups of observations based on their features.

## Objectives

- Explore the dataset and understand its structure.
- Analyze feature distributions using statistical summaries and visualizations.
- Handle missing values and duplicate records.
- Encode categorical variables and standardize features.
- Apply K-Means clustering.
- Use the Elbow Method to help select the number of clusters.
- Evaluate clustering performance using internal evaluation metrics.
- Visualize the resulting clusters.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

### 1. Data Exploration

- Display the first rows, dataset dimensions, column names, and data types.
- Calculate descriptive statistics.
- Visualize age distributions, cholesterol levels by sex, and the relationship between age and cholesterol.
- Inspect missing values and remove duplicate records.

### 2. Data Preprocessing

- Remove the `target` column when it is present.
- Handle missing values.
- Remove duplicate observations.
- Encode categorical features using one-hot encoding.
- Standardize features using `StandardScaler`.

### 3. K-Means Clustering

- Split the processed data into training and testing subsets.
- Apply the Elbow Method to examine different numbers of clusters.
- Fit a K-Means model with three clusters.
- Assign cluster labels to the training and testing data.
- Display the cluster centers.

### 4. Cluster Evaluation

Evaluate the clustering results using:

- **Inertia:** measures the within-cluster sum of squared distances.
- **Silhouette Score:** assesses how well observations fit within their assigned clusters compared with other clusters.
- **Davies-Bouldin Index:** evaluates cluster separation and compactness.

### 5. Visualization

Generate plots to explore feature distributions, select a potential number of clusters, and visualize the clustering results in a two-dimensional projection using the first two standardized features.

## Dataset

The project expects a CSV file named `heart.csv` in the working directory.

The dataset should contain an `age` column and the columns required by the visualizations. If a `target` column exists, it is excluded from the clustering features.

## How to Run

1. Clone this repository.
2. Install the required libraries.
3. Place `heart.csv` in the project directory.
4. Run the Python script.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py
```

## Important Notes

- K-Means is an unsupervised learning algorithm; the clustering results do not directly predict whether a person has heart disease.
- The number of clusters is set to three in this implementation.
- The cluster visualization uses only the first two standardized features, so it does not represent the full feature space.
- The train-test split is used to separate observations, but the current script evaluates clustering metrics on the training subset only.

## Future Improvements

- Compare several values of `k` using multiple evaluation metrics.
- Improve the selection of the number of clusters.
- Visualize clusters using PCA.
- Analyze the characteristics of each cluster.
- Compare clustering results with the dataset's target labels in a separate evaluation step.

## License

This project is available for educational and research purposes. Add a license file if you intend to specify reuse and distribution terms.
