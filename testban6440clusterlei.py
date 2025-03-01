import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ban6440kmeansclusterlei import load_data, perform_kmeans_clustering


class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
            'Relationship.RelationshipType': ['Type1', 'Type2', 'Type3'],
            'Relationship.RelationshipStatus': ['Active', 'Inactive', 'Active'],
            'Relationship.Qualifiers.1.QualifierDimension': ['Dim1', 'Dim2', 'Dim1'],
            'Relationship.Qualifiers.1.QualifierCategory': ['Cat1', 'Cat2', 'Cat1'],
            'Relationship.Quantifiers.1.QuantifierAmount': [100, 200, 300],
            'Relationship.Quantifiers.1.QuantifierUnits': [1, 2, 3],
            'Registration.RegistrationStatus': ['Registered', 'Cancelled', 'Registered'],
            'Registration.ValidationSources': ['Source1', 'Source2', 'Source1']
        })

    def test_load_data(self):
        processed_data = load_data("gleif_goldencopy.csv")
        self.assertIsNotNone(processed_data, "Data loading failed")
        self.assertGreater(processed_data.shape[0], 0, "Dataset is empty after preprocessing")

    def test_kmeans_clustering(self):
        df = self.test_data.copy()

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype("category").cat.codes
        df_scaled = (df - df.mean()) / df.std()

        cluster_labels, kmeans_model = perform_kmeans_clustering(df_scaled, num_clusters=3)

        self.assertEqual(len(cluster_labels), df.shape[0], "Cluster labels count does not match data")
        self.assertIsInstance(kmeans_model, KMeans, "KMeans model not returned correctly")

    def test_empty_dataset(self):
        empty_df = pd.DataFrame()
        result = perform_kmeans_clustering(empty_df, num_clusters=3)
        self.assertIsNone(result, "K-Means should return None for empty datasets")


if __name__ == '__main__':
    unittest.main()
