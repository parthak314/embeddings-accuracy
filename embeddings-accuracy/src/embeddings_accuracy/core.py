from typing import List, Optional
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

class EmbeddingsAccuracy:
    def __init__(self, embeddings: List[List[float]]):
        self.embeddings = embeddings

    def _reduce_dimensions(self, n_components: int = 10) -> List[List[float]]:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_embeddings = svd.fit_transform(self.embeddings)
        return reduced_embeddings

    def _cluster_embeddings(self, n_clusters: int = 5) -> List[int]:
        if n_clusters < 1:
            raise ValueError("Number of clusters must be at least 1.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.embeddings)
        return labels
    
    def _calculate_silhouette_score(self, labels: List[int]) -> float:
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError("At least 2 clusters are required to compute silhouette score.")
        score = float(silhouette_score(self.embeddings, labels))
        return score

    def _knn_neighbors(self, embeddings: Optional[List[List[float]]] = None, k: int = 5) -> List[List[int]]:
        if embeddings is None:
            embeddings = self.embeddings

        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array-like of shape (n_samples, n_features)")

        n = embeddings.shape[0]
        if n == 0:
            return []

        if k < 1:
            raise ValueError("k must be >= 1")

        k = min(k, n - 1)
        distances = np.asarray(pairwise_distances(embeddings, metric="euclidean"))

        d = distances.copy()
        np.fill_diagonal(d, np.inf)

        idx = np.argpartition(d, kth=k-1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        order = np.argsort(d[rows, idx], axis=1)
        neighbors = idx[rows, order]

        return neighbors.tolist()
    
    def _recall_at_k_between(self, other_embeddings: List[List[float]], k: int = 10, sample_size: Optional[int] = 200, seed: int = 42) -> float:
        if len(other_embeddings) != len(self.embeddings):
            raise ValueError("other_embeddings must have the same number of rows as self.embeddings")

        n = len(self.embeddings)

        rng = random.Random(seed)
        indices = list(range(n))
        if sample_size is not None and sample_size < n:
            indices = rng.sample(indices, sample_size)

        gt_neighbors = self._knn_neighbors(k=k)
        other_neighbors = EmbeddingsAccuracy(other_embeddings)._knn_neighbors(k=k)

        recalls = []
        for i in indices:
            gt_set = set(gt_neighbors[i])
            other_set = set(other_neighbors[i])
            if not gt_set:
                recalls.append(0.0)
                continue
            overlap = len(gt_set & other_set)
            recalls.append(overlap / len(gt_set))

        return float(sum(recalls) / len(recalls))
        
    def summarize(self, reduced: Optional[List[List[float]]] = None, labels: Optional[List[int]] = None) -> dict:
        summary = {"n_samples": len(self.embeddings),
                   "dim": len(self.embeddings[0]) if self.embeddings else 0}
        if labels is not None:
            try:
                summary["silhouette_score"] = self._calculate_silhouette_score(labels)
            except Exception as e:
                summary["silhouette_error"] = str(e)

        if reduced is not None:
            try:
                summary["knn_recall_at_10_vs_reduced"] = self._recall_at_k_between(reduced, k=10)
            except Exception as e:
                summary["knn_recall_error"] = str(e)

        return summary

    @staticmethod
    def get_accuracy(embeddings_or_self: Optional[object] = None, n_components: int = 32, n_clusters: int = 5, k: int = 10, sample_size: Optional[int] = 200, seed: int = 42) -> dict:
        if isinstance(embeddings_or_self, EmbeddingsAccuracy):
            ea = embeddings_or_self
            embeddings = getattr(ea, "embeddings", None)
        else:
            embeddings = embeddings_or_self
            ea = EmbeddingsAccuracy(embeddings)

        if embeddings is None:
            raise ValueError("embeddings must be provided either by calling on an instance or as the first argument")

        out = {"n_samples": len(embeddings), "dim": len(embeddings[0]) if embeddings else 0}

        try:
            reduced = ea._reduce_dimensions(n_components=n_components)
            out["reduced_dim"] = len(reduced[0]) if reduced is not None and len(reduced) > 0 else 0
        except Exception as e:
            reduced = None
            out["reduce_error"] = str(e)

        try:
            labels = ea._cluster_embeddings(n_clusters=n_clusters)
        except Exception as e:
            labels = None
            out["cluster_error"] = str(e)

        if labels is not None:
            try:
                out["silhouette_score"] = ea._calculate_silhouette_score(labels)
            except Exception as e:
                out["silhouette_error"] = str(e)

        if reduced is not None:
            try:
                out["knn_recall_at_k_vs_reduced"] = ea._recall_at_k_between(reduced, k=k, sample_size=sample_size, seed=seed)
            except Exception as e:
                out["knn_recall_error"] = str(e)

        return out