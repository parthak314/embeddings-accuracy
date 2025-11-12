"""
Small demo that runs the main flows with synthetic data and prints a summary of the results using the embeddings_accuracy library.
"""

from embeddings_accuracy.core import EmbeddingsAccuracy
import json
import random

def generate_random_embeddings(n: int = 1000, dim: int = 256, seed: int = 42) -> list:
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(dim)] for _ in range(n)]

def main():
    print("Running demo with synthetic embeddings (small)")
    emb = generate_random_embeddings(n=500, dim=128)
    
    try:
        summary = EmbeddingsAccuracy.get_accuracy(
            emb,
            n_components=32,
            n_clusters=5,
            k=10,
            sample_size=200,
            seed=42,
        )
    except Exception as e:
        summary = {"error": str(e)}

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()