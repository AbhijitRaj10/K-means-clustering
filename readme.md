Documentation: Customer Segmentation using K-Means Clustering
1. Importing Necessary Libraries
python
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pandas: For data manipulation and analysis.
numpy: For numerical operations.
seaborn: For statistical data visualization.
matplotlib: For plotting graphs.
warnings: To ignore warnings during execution.
2. Loading the Dataset
python
 
df = pd.read_csv('data/Mall_Customers.csv', index_col=0, header=0)
df.head()
Load the dataset Mall_Customers.csv and display the first few rows using head().
3. Data Exploration
Gender Distribution
python
 
df['Gender'].value_counts()
Count the number of male and female customers.
Gender Distribution Pie Chart
python
 
data = [112, 88]
keys = ['Female', 'Male']
palette_color = sns.color_palette('bright')[0:5]
explode = [0.1, 0]

plt.pie(data, labels=keys, colors=palette_color, explode=explode, autopct='%.2f%%')
plt.show()
Plot a pie chart to visualize the gender distribution among customers.
Gender Distribution Count Plot
python
 
sns.countplot(x='Gender', data=df)
plt.show()
Create a count plot to visualize the number of male and female customers.
Age Distribution by Gender
python
 
sns.barplot(x="Gender", y="Age", data=df)
plt.show()
Plot a bar chart showing the average age of male and female customers.
Age Distribution Histogram
python
 
sns.histplot(data=df, x='Age', hue="Gender", bins=6, multiple='dodge')
plt.show()
Plot a histogram to visualize the age distribution of customers, differentiated by gender.
Annual Income Distribution Histogram
python
 
sns.histplot(data=df, x='Annual Income (k$)', hue="Gender", bins=6, multiple='dodge')
plt.show()
Plot a histogram to visualize the annual income distribution of customers, differentiated by gender.
Spending Score Distribution Histogram
python
 
sns.histplot(data=df, x='Spending Score (1-100)', hue="Gender", bins=6, multiple='dodge')
plt.show()
Plot a histogram to visualize the spending score distribution of customers, differentiated by gender.
4. Data Information and Summary
python
 
print(df.shape)
print(df.info())
print(df.isnull().sum())
Display the shape of the DataFrame.
Display detailed information about the DataFrame.
Check for any missing values in the DataFrame.
Descriptive Statistics
python
 
df.describe(include='all')
Display descriptive statistics for all columns in the DataFrame.
5. Data Preparation for Clustering
python
 
df.columns
Display column names.
Boxplot for Annual Income
python
 
df.boxplot("Annual Income (k$)")
plt.show()
Create a boxplot to visualize the distribution of annual income.
6. Extracting Relevant Features
python
 
X = df.iloc[:, 2:4]  # Extracting 'Annual Income' and 'Spending Score' columns
Extract the 'Annual Income' and 'Spending Score' columns for clustering.
7. K-Means Clustering
python
 
from sklearn.cluster import KMeans

# Finding the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying KMeans to the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X.iloc[y_kmeans == 3, 0], X.iloc[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X.iloc[y_kmeans == 4, 0], X.iloc[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
Import the KMeans class from scikit-learn.
Use the elbow method to find the optimal number of clusters.
Apply KMeans clustering to the dataset with the optimal number of clusters.
Visualize the clusters and their centroids.
Conclusion
This code provides a comprehensive approach to customer segmentation using K-means clustering. It involves loading the dataset, exploring the data, preparing it for clustering, and finally, applying and visualizing the results of K-means clustering.