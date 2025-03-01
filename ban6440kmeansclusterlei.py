import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

file_path = "gleif_goldencopy.csv"
df = pd.read_csv(file_path)

selected_columns = [
    'Relationship.EndNode.NodeIDType', 'Relationship.RelationshipType', 'Relationship.RelationshipStatus',
    'Relationship.Period.1.periodType', 'Relationship.Period.2.periodType', 'Relationship.Period.3.periodType',
    'Relationship.Period.4.periodType', 'Relationship.Period.5.periodType', 'Relationship.Qualifiers.1.QualifierDimension',
    'Relationship.Qualifiers.1.QualifierCategory', 'Relationship.Qualifiers.2.QualifierDimension',
    'Relationship.Qualifiers.2.QualifierCategory', 'Relationship.Qualifiers.3.QualifierDimension',
    'Relationship.Qualifiers.3.QualifierCategory', 'Relationship.Qualifiers.4.QualifierDimension',
    'Relationship.Qualifiers.4.QualifierCategory', 'Relationship.Qualifiers.5.QualifierDimension',
    'Relationship.Qualifiers.5.QualifierCategory', 'Relationship.Quantifiers.1.MeasurementMethod',
    'Relationship.Quantifiers.1.QuantifierAmount', 'Relationship.Quantifiers.1.QuantifierUnits',
    'Relationship.Quantifiers.2.MeasurementMethod', 'Relationship.Quantifiers.2.QuantifierAmount',
    'Relationship.Quantifiers.2.QuantifierUnits', 'Relationship.Quantifiers.3.MeasurementMethod',
    'Relationship.Quantifiers.3.QuantifierAmount', 'Relationship.Quantifiers.3.QuantifierUnits',
    'Relationship.Quantifiers.4.MeasurementMethod', 'Relationship.Quantifiers.4.QuantifierAmount',
    'Relationship.Quantifiers.4.QuantifierUnits', 'Relationship.Quantifiers.5.MeasurementMethod',
    'Relationship.Quantifiers.5.QuantifierAmount', 'Relationship.Quantifiers.5.QuantifierUnits',
    'Registration.RegistrationStatus', 'Registration.ManagingLOU', 'Registration.ValidationSources',
    'Registration.ValidationDocuments', 'Registration.ValidationReference'
]

df = df[selected_columns]

df.fillna(df.mode().iloc[0], inplace=True)


label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

inertia = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Determine Optimal k")
plt.show()

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

silhouette_avg = silhouette_score(df_scaled, df["Cluster"])
print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.4f}")

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df["Cluster"], palette="viridis")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering Visualization (PCA)")
plt.legend(title="Cluster")
plt.show()

df = pd.read_csv(file_path, dtype=str, low_memory=False)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

df = pd.read_csv(file_path, dtype=str, na_values=["", " ", "NULL", "NaN"], low_memory=False)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

print("Missing values after imputation:", df.isnull().sum().sum())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

silhouette_avg = silhouette_score(df_scaled, df["Cluster"])
print(f"Silhouette Score: {silhouette_avg:.4f}")

print("Missing values in dataset before scaling:")
print(df.isnull().sum().sum())
