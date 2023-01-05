from typing import List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

def find_clustering(D: np.ndarray, cn: np.ndarray,  min_coverage: float) -> List[int]:
    print(f"Generating clusters from {D.shape[0]} cells with min coverage {min_coverage}")
    coverage = np.mean(D)
    cell_to_cluster = {cell: cell for cell in range(D.shape[0])}
    while coverage < min_coverage and D.shape[0] > 1:
        print(f"Coverage {coverage} clusters {len(set(cell_to_cluster.values()))}")
        clusters = int(coverage / min_coverage * cn.shape[0])
        clustering = AgglomerativeClustering(n_clusters=clusters, metric='euclidean').fit(cn)
        clustering = list(clustering.labels_)
        D = cluster_array(D, clustering, function="sum")
        cn = cluster_array(cn, clustering, function="mean")
        cn = cn.astype(int)
        coverage = np.mean(D)
        cell_to_cluster = {cell: clustering[old_c] for cell, old_c in cell_to_cluster.items()}

    print(f"Cluster generation complete with {len(set(cell_to_cluster))} clusters.")
    res = [(cell, c) for cell, c in cell_to_cluster.items()]
    res.sort()
    return [r[1] for r in res]

def cluster_array(a: np.ndarray, labels: List[int], function: str = "mean"):
    clustered = np.zeros((len(set(labels)), a.shape[1]))
    for cluster in set(labels):
        rows = [i for i, l in enumerate(labels) if l == cluster]
        if function == "mean":
            m = a[rows, :].mean(axis=0)
        elif function == "sum":
            m = a[rows, :].sum(axis=0)
        elif function == "median":
            m = np.median(a[rows, :], axis=0)
        else:
            raise RuntimeError(f"Unknown function {function}")
        clustered[cluster, :] = m
    return clustered