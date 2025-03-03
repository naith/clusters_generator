# Clusters Generator

## Project Description
Clusters Generator is a tool designed to generate artificial datasets with cluster structures. It is intended for testing and analyzing clustering algorithms and their applications in various fields, such as machine learning and data processing.

## Features
- Generate synthetic clustered datasets with adjustable parameters.
- Support for various data distribution types.
- Visualization of generated data.
- Export data to commonly used formats (e.g., CSV, JSON).
- **Algorithm Testing**: Includes tests for various clustering algorithms to evaluate their performance on generated datasets.

## Installation
The project can be downloaded and run locally. The following dependencies are required:

### Requirements
- Python 3.x
- Libraries: numpy, matplotlib, scikit-learn (for visualization), hdbscan (for HDBSCAN algorithm), and any other required libraries.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/naith/clusters_generator.git
   cd clusters_generator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Example usage of the tool:
```python
from clusters_generator import generate_clusters

data = generate_clusters(n_clusters=3, n_samples=300)
# Further manipulation of the generated data
```

## Algorithm Tests
The repository includes tests for various clustering algorithms to assess their performance on the generated datasets. The following test scripts are available:

- `k_nn_test.py`: Tests the k-Nearest Neighbors algorithm.
- `dbscan_test.py`: Tests the DBSCAN algorithm.
- `hdbscan_test.py`: Tests the HDBSCAN algorithm.
- `mean_shift_test.py`: Tests the Mean Shift algorithm.
- `optics_test.py`: Tests the OPTICS algorithm.
- `spectral_test.py`: Tests the Spectral Clustering algorithm.

To run a test, execute the corresponding script. For example:
```bash
python dbscan_test.py
```

## Contributing
If you wish to contribute to the project, feel free to create a pull request or report an issue in the issues section.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
If you have any questions, contact me on GitHub or open an issue in this repository.

