"""
Command-line interface for the embeddings_accuracy library.
This module allows users to interact with the library's functionality directly from the terminal.
"""

import argparse
import json
from embeddings_accuracy.core import EmbeddingsAccuracy
from embeddings_accuracy.utils import _generate_random_embeddings

def main():
    parser = argparse.ArgumentParser(description="Embeddings Accuracy CLI")
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for random embeddings')
    parser.add_argument('--dim', type=int, default=256, help='Dimensionality of the embeddings')
    parser.add_argument('--n_components', type=int, default=32, help='Number of components for dimensionality reduction')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for K-means')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors for recall@k')
    parser.add_argument('--sample_size', type=int, default=200, help='Sample size for recall computation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    embeddings = _generate_random_embeddings(n=args.n_samples, dim=args.dim, seed=args.seed)
    summary = EmbeddingsAccuracy.get_accuracy(
        embeddings,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        k=args.k,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()