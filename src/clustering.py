import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
from typing import List
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TitleClustering:

    def __init__(
        self, device: torch.device, model_name: str, titles: pd.Series
    ) -> None:
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.titles = titles.to_list()

    def encode(self, batch_size: int = 16) -> np.ndarray:
        embeddings = []

        for i in tqdm(
            range(0, len(self.titles), batch_size),
            desc="Processing batches",
            unit="batch",
        ):
            batch = self.titles[i : i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device,
            )

            embeddings.extend(batch_embeddings.cpu().numpy())

        return np.array(embeddings)

    def elbow_method(
        self, embeddings: np.ndarray, max_clusters: int = 10
    ) -> List[float]:
        """
        Runs K-Means clustering for different values of k and plots the inertia (WCSS) to find the elbow point.

        :param embeddings: The feature embeddings for clustering.
        :param max_clusters: The maximum number of clusters to test.
        """
        wcss = []
        cluster_range = range(1, max_clusters + 1)

        for k in tqdm(cluster_range, desc="Running K-Means", unit="cluster"):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            wcss.append(kmeans.inertia_)

        return wcss

    def reduce_dimensions(self, embeddings, n_components: int = 2) -> np.ndarray:
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        pca = PCA(n_components)
        reduced_embeddings = pca.fit_transform(scaled_embeddings)
        return reduced_embeddings

    def clustering(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters

    def combine_results(
        self, features: np.ndarray, clusters: np.ndarray
    ) -> pd.DataFrame:

        df = pd.DataFrame({"product_title": self.titles, "cluster": clusters})

        for i in range(features.shape[1]):
            df[f"dim_{i}"] = features[:, i]

        return df
