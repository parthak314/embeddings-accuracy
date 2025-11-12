# Embeddings Accuracy

## Overview
The `embeddings-accuracy` library provides tools for evaluating the quality of embeddings in terms of their ability to separate different clusters and semantically group similar items. It includes functionality for dimensionality reduction, clustering, silhouette score calculation, k-NN neighbor retrieval, and recall computation.

## Features
- Dimensionality reduction using PCA/SVD
- Clustering with K-means
- Silhouette score calculation for cluster evaluation
- k-NN neighbor retrieval
- Recall computation between different sets of embeddings

## Installation
To install the required dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Usage
Here is a simple example of how to use the library:

```python
from embeddings_accuracy import EmbeddingsAccuracy
import numpy as np

# Generate random embeddings
embeddings = np.random.rand(1000, 256).tolist()

# Create an instance of EmbeddingsAccuracy
ea = EmbeddingsAccuracy(embeddings)

# Get accuracy metrics
metrics = ea.get_accuracy(n_components=32, n_clusters=5, k=10)
print(metrics)
```

## Running Tests
To run the tests for the library, you can use the following command:

```bash
pytest tests/
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.