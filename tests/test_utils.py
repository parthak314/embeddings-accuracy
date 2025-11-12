from typing import List
import random

def generate_random_embeddings(n: int = 1000, dim: int = 256, seed: int = 42) -> List[List[float]]:
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(dim)] for _ in range(n)]

def test_generate_random_embeddings():
    embeddings = generate_random_embeddings(n=10, dim=5, seed=42)
    assert len(embeddings) == 10
    assert all(len(embedding) == 5 for embedding in embeddings)

def test_generate_random_embeddings_different_seeds():
    embeddings1 = generate_random_embeddings(n=10, dim=5, seed=42)
    embeddings2 = generate_random_embeddings(n=10, dim=5, seed=43)
    assert embeddings1 != embeddings2

def test_generate_random_embeddings_same_seed():
    embeddings1 = generate_random_embeddings(n=10, dim=5, seed=42)
    embeddings2 = generate_random_embeddings(n=10, dim=5, seed=42)
    assert embeddings1 == embeddings2