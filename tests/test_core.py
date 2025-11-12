from embeddings_accuracy import EmbeddingsAccuracy
import numpy as np
import pytest

class TestEmbeddingsAccuracy:
    @pytest.fixture
    def setup_embeddings(self):
        self.embeddings = np.random.rand(100, 256).tolist()
        self.ea = EmbeddingsAccuracy(self.embeddings)

    def test_reduce_dimensions(self, setup_embeddings):
        reduced = self.ea._reduce_dimensions(n_components=10)
        assert len(reduced) == len(self.embeddings)
        assert len(reduced[0]) == 10

    def test_cluster_embeddings(self, setup_embeddings):
        labels = self.ea._cluster_embeddings(n_clusters=5)
        assert len(labels) == len(self.embeddings)
        assert len(set(labels)) == 5

    def test_calculate_silhouette_score(self, setup_embeddings):
        labels = self.ea._cluster_embeddings(n_clusters=5)
        score = self.ea._calculate_silhouette_score(labels)
        assert score >= -1
        assert score <= 1

    def test_knn_neighbors(self, setup_embeddings):
        neighbors = self.ea._knn_neighbors(k=5)
        assert len(neighbors) == len(self.embeddings)
        assert all(len(n) == 5 for n in neighbors)

    def test_recall_at_k_between(self, setup_embeddings):
        other_embeddings = np.random.rand(100, 256).tolist()
        recall = self.ea._recall_at_k_between(other_embeddings, k=10)
        assert 0 <= recall <= 1

    def test_summarize(self, setup_embeddings):
        labels = self.ea._cluster_embeddings(n_clusters=5)
        summary = self.ea.summarize(labels=labels)
        assert "n_samples" in summary
        assert "dim" in summary
        assert "silhouette_score" in summary

    def test_get_accuracy(self, setup_embeddings):
        accuracy = EmbeddingsAccuracy.get_accuracy(self.embeddings, n_components=10, n_clusters=5, k=10)
        assert "silhouette_score" in accuracy
        assert "knn_recall_at_k_vs_reduced" in accuracy
        assert accuracy["n_samples"] == len(self.embeddings)