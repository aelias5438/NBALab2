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
cluster_cols = merged_data[['PTS Per Minute', 'Trp-Dbl', 'TRB', 'AST', 'MP']]
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
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue=cluster_labels, palette='Set1')
# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['AST'], hue=cluster_labels, palette='Set1')
# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['MP'], hue=cluster_labels, palette='Set1')


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
# Additionally, we have a not great variance explained of .42. This shows under half of the variation is explained
# By the clustering. This is not as high as we would like it to be.
# To see if we can improve this, let's experiment by changing k and seeing if we can improve results
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

# I like k=6 because it has the best silhouette score outside of 2 and the elbow method shows a significant drop in inertia at that point. We aren't going with k=2 because as 
# k increases from 2 we see large decreases in inertia Let's cluster with k=6 and see how it looks.
# %%
kmeans = KMeans(n_clusters=6, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# %%
cluster_labels = kmeans.labels_

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['TRB'], hue=cluster_labels, palette='Set1')
# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['AST'], hue=cluster_labels, palette='Set1')
# %%

# %%
sns.relplot(x=cluster_cols['PTS Per Minute'], y=cluster_cols['MP'], hue=cluster_labels, palette='Set1')

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

print("Variance Explained (k=6):", variance_explained)

# Variance explained of .71 is very good, showing that our clusters have been effective
# as 71% of the variation is explained by our clusters. Additionally, our silhouette score
# has decreased to .3, showing there is more overlap. This is a tradeoff we have to make
# When increasing the number of clusters, as in a data set like this one
# In the nba there is no actual cutoff between good and bad, so it is therefore hard to have
# a very high silhoutte score, and in context .30 is pretty strong due to the nature of the data.
# Overall, k=6 is much better than the original k=3.

# %% Pick Players
# Fill cluster assignments into the original merged_data
merged_data.loc[cluster_cols.index, 'Cluster'] = cluster_labels


# Compare salary to average salary of cluster for each player
merged_data['Cluster_Avg_Salary'] = merged_data.groupby('Cluster')['Salary'].transform('mean')

# Find difference between cluster and individuals, if this number is negative that means they are good for the cluster they are in
merged_data['Salary_Difference'] = merged_data['Salary'] - merged_data['Cluster_Avg_Salary']

# This ensures that only active players getting real minutes are considered
min_minutes = 800
analysis_df = merged_data[merged_data['MP'] >= min_minutes].copy()

# pick our 10 best by looking at the largest gap between salary and cluster average salary.
best_deals = (analysis_df.sort_values('Salary_Difference')[['Player','Cluster','Salary','Cluster_Avg_Salary','Salary_Difference','PTS Per Minute','TRB','AST','MP']].head(10))

best_deals.head(10)
# Looking at our eda, clusters 1, 2, and 3 are our high performance clusters, and 
# we would be happy with any of them. Therefore, to grab a top 3 value picks, 
# I'm looking for just the best 3 deals among those clusters,
#  which we have found. Here are the players in order:
# 1. Jamal Shead 2. Brandon Williams 3.Collin Gillepsie. 
# %% Do the same but for the worst 10 finding the most overrated players

overpaid = (analysis_df.sort_values('Salary_Difference', ascending=False)[['Player','Cluster','Salary','Cluster_Avg_Salary','Salary_Difference','PTS Per Minute','TRB','AST','MP']].head(10))
overpaid.head(10)
# Since we found before clusters 1, 2, and 3 were our high performance clusters, 
# We will look there for our worst deals in those cluster.
# The most overpaid options are 1. Stephen Curry 2. Joel Embiid. 3. Jaylen Brown             
# %% Mid options
best_deals_0_4 = analysis_df[analysis_df['Cluster'].isin([0,4])].sort_values('Salary_Difference').groupby('Cluster').head(10)[['Player','Cluster','Salary','Cluster_Avg_Salary','Salary_Difference','PTS Per Minute','TRB','AST','MP']]
best_deals_0_4.head(10)
# If we cant get anyone in a high performance cluster, we should look to clusters 4 and 0 as our
# Mid tier options. The 3 most underpaid players in those clusters are:
# 1. Maxime Raynaud 2. Toumani Camara 3. Justin Champagnie


# %%
