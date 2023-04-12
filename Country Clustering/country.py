import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Veri setini yükleyin
data = pd.read_csv("Country-data.csv")

# Veri setindeki özellikleri seçin
X = data[['child_mort', 'income', 'gdpp']]

# Verileri normalize edin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method ile en uygun küme sayısını belirleyin
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)
plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# K-means algoritmasını kullanarak verileri kümeleyin
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Her ülkenin hangi kümede olduğunu belirleyin
clusters = kmeans.predict(X_scaled)

# Küme merkezlerini elde edin
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Verileri görselleştirin
plt.scatter(data['income'], data['child_mort'], c=clusters)
plt.scatter(centers[:, 1], centers[:, 0], marker='*', s=300,
            c='red', label='Cluster centers')
plt.xlabel('Income')
plt.ylabel('Child Mortality')
plt.legend()
plt.show()
