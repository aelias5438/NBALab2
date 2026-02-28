#%%
import pandas as pd
import numpy as np 
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

np.random.seed(0)

#%%
# Load the data
nbadata = pd.read_csv('nba_2025.txt', sep = ',', encoding = 'latin-1')
salary = pd.read_csv('2025_salaries.csv', header = 1, encoding = 'latin-1')

print(salary.head())

# Merge Data
merged_data = pd.merge(salary, nbadata, on='Player')

duplicates = merged_data[merged_data.duplicated(subset='Player',keep=False)]
# print(merged_data)

# %% Start data cleaning
merged_data = merged_data.drop_duplicates(subset='Player')

# %%
# Convert to ppm so we can compare fairly
merged_data['PTS Per Minute']= merged_data['PTS'] / merged_data['MP']
# Rename column to salary
merged_data = merged_data.rename(columns={'2025-26': 'Salary'})
# Look at dtypes to see if we have to fix any before clustering
merged_data.dtypes
merged_data['Salary'] = (merged_data['Salary'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float))

# Pick columns for clustering
cluster_cols = merged_data[['PTS Per Minute', 'Salary', "FT%", 'TRB']]
cluster_cols = cluster_cols.dropna()


# %% Standardize the data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(cluster_cols)

# %% Cluster
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

#%%

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['Salary'], hue=cluster_labels, palette='Set1')
# Cluster 2 is the high salary players, while the other two are pretty similiar
# %%
# sns.relplot(x=cluster_cols['FT%'], y=cluster_cols['Salary'], hue=cluster_labels, palette='Set1')
# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue=cluster_labels, palette='Set1')
# Unsuprisingly, the high salary players have the best stats

# %% Evaluate the clustering
sil_score = silhouette_score(X_scaled, cluster_labels)
print("Silhouette Score:", sil_score)
# Total Sum of Squares (TSS)
X_mean = np.mean(X_scaled, axis=0)
tss = np.sum((X_scaled - X_mean) ** 2)

# Within-cluster sum of squares (WCSS)
wcss = kmeans.inertia_

# Variance Explained
variance_explained = 1 - (wcss / tss)

print("Variance Explained (k=3):", variance_explained)
# Low silhoutte indicates that our clusters have a lot of overlap which is not a great sign.
# Additioanally, we have a not great variance explained of .47. This shows under half of the variation is explained
# By the clustering. This is not as high as we would like it to be.
# To see if we can improve this, let's expeeriment by changing k and seeing if we can improve results
# %% 
inertias = []
sil_scores = []

for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X_scaled)
    
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
# %% Elbow method to pick k
plt.plot(range(2, 10), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# %% Check Silhoutte
plt.plot(range(2, 10), sil_scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()

# I like k=4 because it has a good silhouette score and the elbow method shows a significant drop in inertia at that point. Let's cluster with k=4 and see how it looks.
# %%
kmeans = KMeans(n_clusters=4, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# %%
cluster_labels = kmeans.labels_

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue= cluster_cols['Salary'],    palette='viridis', style =cluster_labels )
# Not exactly sure, but 3 is definitely the higher salary players, while the other 3 are pretty similiar. 1 has a few diamonds in the rough, with a clustering at a high average of about .5 ppm and the lowest salary
# %%
# sns.relplot(x=cluster_cols['FT%'], y=cluster_cols['Salary'], hue=cluster_labels, palette='Set1')

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue=cluster_labels, palette='Set1')

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['FT%'], hue=cluster_labels, palette='Set1')

# %%
sil_score = silhouette_score(X_scaled, cluster_labels)
print("Silhouette Score:", sil_score)
# Total Sum of Squares (TSS)
X_mean = np.mean(X_scaled, axis=0)
tss = np.sum((X_scaled - X_mean) ** 2)

# Within-cluster sum of squares (WCSS)
wcss = kmeans.inertia_

# Variance Explained
variance_explained = 1 - (wcss / tss)

print("Variance Explained (k=4):", variance_explained)

# %% I liked what I saw in the earlier graph, as it showed a few lower salary players high in TRB and pts
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue= cluster_cols['Salary'],    palette='viridis', style =cluster_labels )

# Thresholds based on distribution
ppm_cutoff = merged_data['PTS Per Minute'].quantile(0.75)
trb_cutoff = merged_data['TRB'].quantile(0.75)

print("PPM cutoff:", ppm_cutoff)
print("TRB cutoff:", trb_cutoff)
