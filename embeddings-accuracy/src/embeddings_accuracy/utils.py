from typing import List
import random

def generate_random_embeddings(n: int = 1000, dim: int = 256, seed: int = 42) -> List[List[float]]:
    rnd = random.Random(seed)
    return [[rnd.random() for _ in range(dim)] for _ in range(n)]