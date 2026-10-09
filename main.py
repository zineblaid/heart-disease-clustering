# Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Chargement des données
df = pd.read_csv("heart.csv", encoding="latin1")  # Assurez-vous que le fichier est bien CSV


# PART 1 – DATA EXPLORATION
# 1. Download the Heart Disease dataset and load it into your code using Pandas
#2. Display first rows, dataset shape, column names, and data types
print("=== 5 premières lignes ===")
print(df.head(), "\n")

print("=== Dimensions du dataset ===")
print(df.shape, "\n")

print("=== Colonnes et types ===")
print(df.dtypes, "\n")

# 3. Remove the target column
if 'target' in df.columns:
    df_features = df.drop('target', axis=1)
else:
    df_features = df.copy()
    print("Dataset after removing the target column:")
print(df_features.head())

# 4. Compute basic statistics
print("=== Statistiques de base ===")
print(df_features.describe(), "\n")
# 5. Analyze feature distributions and visualize
# -----------------
plt.figure(figsize=(12, 8))

# Histogram of age
plt.subplot(2, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue')
plt.title("Distribution of Age")

# Boxplot of cholesterol by sex
plt.subplot(2, 2, 2)
sns.boxplot(x='sex', y='chol', data=df)
plt.title("Cholesterol by Sex")

# Scatter plot: Age vs Cholesterol
plt.subplot(2, 2, 3)
plt.scatter(df['age'], df['chol'], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Age vs Cholesterol")

plt.tight_layout()
plt.show()

# # 6. Identify missing values and remove them if any
print("\n=== Missing values in each column ===")
print(df_features.isnull().sum())
df_features = df_features.dropna()  # Remove missing rows if any
print("Missing values removed.")
# Suppression des doublons
df_features = df_features.drop_duplicates()
# 8. Check for duplicates and remove them
print("\nNumber of duplicates before removal:", df_features.duplicated().sum())
df_features = df_features.drop_duplicates()
print("Duplicates removed. Number of rows after:", df_features.shape[0])
## 9. Encode categorical variables (One-Hot Encoding)
categorical_cols = df_features.select_dtypes(include=['object']).columns
df_features = pd.get_dummies(df_features, columns=categorical_cols)

# Normalisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)
# 10 & 11. Select features and normalize/standardize numerical features

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)
print("\nFeatures normalized. Ready for clustering.")


# Graphiques (tout dans une seule fenêtre)

plt.figure(figsize=(12, 8))

# Graphique 1 : histogramme de l'âge
plt.subplot(2, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue')
plt.title("Distribution des âges")

# Graphique 2 : boxplot du cholestérol par sexe
plt.subplot(2, 2, 2)
sns.boxplot(x='sex', y='chol', data=df)
plt.title("Cholestérol par sexe")

# Graphique 3 : scatter plot âge vs cholestérol
plt.subplot(2, 2, 3)
plt.scatter(df['age'], df['chol'], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Cholestérol")
plt.title("Age vs Cholestérol")

plt.tight_layout()
plt.show()


# PARTIE 2 : K-Means

# Split (80% train, 20% test)
X_train, X_test = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Elbow Method pour choisir k
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Méthode du coude pour choisir k')
plt.show()

# On choisit k=3 par exemple
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train)

# Labels
labels_train = kmeans.labels_
labels_test = kmeans.predict(X_test)

# Centres des clusters
print("=== Centres des clusters ===")
print(kmeans.cluster_centers_, "\n")


# PARTIE 3 : Évaluation

silhouette = silhouette_score(X_train, labels_train)
davies = davies_bouldin_score(X_train, labels_train)
print("Inertia (Within-cluster sum of squares):", kmeans.inertia_)
print(f"Silhouette Score: {silhouette}")
print(f"Davies-Bouldin Index: {davies}")

# Visualisation des clusters (2 premières dimensions)
plt.figure(figsize=(6,6))
plt.scatter(X_train[:,0], X_train[:,1], c=labels_train, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', marker='X', s=200)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clusters K-Means")
plt.show()
