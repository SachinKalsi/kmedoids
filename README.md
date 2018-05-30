# K-Medoids 
K-Medoids is a clustering algorithm. **Partitioning Around Medoids (PAM)** algorithm is one such implementation of K-Medoids

## Prerequisites
  - Scipy
  - Numpy
 
## Getting Started
`from KMedoids import KMedoids`

## Parameters
- `n_cluster`: number of clusters
- `max_iter`: maximum number of iterations
- `tol`: tolerance level

## Example 
[Wiki Example](https://en.wikipedia.org/wiki/K-medoids#Demonstration_of_PAM)
```
data = [[2, 6], [3, 4], [3, 8], [4, 7], [6, 2], [6, 4], [7, 3], [7, 4], [8, 5], [7, 6]]
k_medoids = KMedoids(n_cluster=2)
k_medoids.fit(data)
```
#### Visualization
[Clusters](./demo.ipynb)
