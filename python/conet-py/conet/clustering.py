from typing import List

import numpy as np
import scipy.cluster.hierarchy as sch

def find_clustering(D: np.ndarray, cn: np.ndarray,  t_min: float, t_max: float) -> List[int]:
    x = sch.linkage(cn, method = 'ward')
    linkage_cluster_to_cells = {cell: [cell] for cell in range(cn.shape[0])}
    n = cn.shape[0]

    for i in range(x.shape[0]):
        left_clust = x[i,0]
        right_clust = x[i, 1]
        assert left_clust < i +n and right_clust < i + n
        linkage_cluster_to_cells[i + n] = linkage_cluster_to_cells[left_clust] + linkage_cluster_to_cells[right_clust]
        assert len(linkage_cluster_to_cells[i + n]) == x[i, 3]

    def get_cluster_coverage(cells: List[int]) -> float:
        return np.sum(D[cells, :]) / D.shape[1]

    final_clusters = {}
    cluster_id = 0
    full_coverage = get_cluster_coverage(linkage_cluster_to_cells[2*n -2])
    if full_coverage < t_min:
        raise RuntimeError(f"Full cluster has coverage {full_coverage} < min={t_min}")

    def split_cluster(i: int) -> None:
        nonlocal cluster_id
        coverage = get_cluster_coverage(linkage_cluster_to_cells[i +n])
        assert coverage >= t_min
        if len(linkage_cluster_to_cells[i +n]) == 1:
            final_clusters[cluster_id] = linkage_cluster_to_cells[i +n]
            cluster_id += 1
            return
        left_clust = int(x[i, 0]) - n
        right_clust = int(x[i, 1]) - n
        assert left_clust < D.shape[0]
        assert right_clust < D.shape[0]
        left_clust_cov = get_cluster_coverage(linkage_cluster_to_cells[left_clust + n])
        right_clust_cov = get_cluster_coverage(linkage_cluster_to_cells[right_clust + n])

        if left_clust_cov < t_min and right_clust_cov < t_max:
            print(f"1 {left_clust_cov} with cells {len(linkage_cluster_to_cells[left_clust + n])}   {right_clust_cov} with cells {len(linkage_cluster_to_cells[right_clust + n])}")
            final_clusters[cluster_id] = linkage_cluster_to_cells[i + n]
            cluster_id += 1
            return
        if left_clust_cov < t_min and right_clust_cov >= t_max:
            print(
                f"2 {left_clust_cov} with cells {len(linkage_cluster_to_cells[left_clust + n])}   {right_clust_cov} with cells {len(linkage_cluster_to_cells[right_clust + n])}")

            split_cluster(right_clust)
            for cell in linkage_cluster_to_cells[left_clust + n]:
                final_clusters[cluster_id] = [cell]
                cluster_id += 1
            return
        if right_clust_cov < t_min and left_clust_cov < t_max:
            print(
                f"3 {left_clust_cov} with cells {len(linkage_cluster_to_cells[left_clust + n])}   {right_clust_cov} with cells {len(linkage_cluster_to_cells[right_clust + n])}")

            final_clusters[cluster_id] = linkage_cluster_to_cells[i + n]
            cluster_id += 1
            return
        if right_clust_cov < t_min and left_clust_cov >= t_max:
            print(
                f"4 {left_clust_cov} with cells {len(linkage_cluster_to_cells[left_clust + n])}   {right_clust_cov} with cells {len(linkage_cluster_to_cells[right_clust + n])}")

            split_cluster(left_clust)
            for cell in linkage_cluster_to_cells[right_clust + n]:
                final_clusters[cluster_id] = [cell]
                cluster_id += 1
            return
        split_cluster(left_clust)
        split_cluster(right_clust)
        return

    split_cluster(n - 2)
    cell_cluster_pairs = []
    for cluster, cells in final_clusters.items():
        cell_cluster_pairs.extend([(c, cluster) for c in cells])
    cell_cluster_pairs.sort()
    assert len(cell_cluster_pairs) == D.shape[0]
    assert cell_cluster_pairs[0][0] == 0
    print(f"Clustering complete with {len(final_clusters)} clusters.")
    for cluster, cells in final_clusters.items():
        print(f"Cluster {cluster} with {len(cells)} cells")
    return [x for _,x in cell_cluster_pairs]

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