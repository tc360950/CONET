from collections import defaultdict
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt

import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

def find_clustering_top_down(D: np.ndarray, cn: np.ndarray, t_min: float, t_max: float, data_dir: Path, cn_for_snvs:np.ndarray) -> List[int]:
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

    clustering = [x for _, x in cell_cluster_pairs]
    """
    PLOTTING 
    """
    colors = []
    for i in range(len(cell_cluster_pairs)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    dflt_col = "#808080"  # Unclustered gray
    link_cols = {}

    for i, i12 in enumerate(x[:, :2].astype(int)):
        c1, c2 = (link_cols[y] if y > len(x) else colors[clustering[y]]
                  for y in i12)
        link_cols[i + 1 + len(x)] = c1 if c1 == c2 else dflt_col

    plt.figure(figsize=(50, 50))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    sch.dendrogram(
        x,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        link_color_func=lambda x: link_cols[x],
    )
    plt.savefig(f'{str(data_dir / Path("dendogram"))}.png')
    return [x for _,x in cell_cluster_pairs]

















def find_clustering_top_down_cn_normalization(D: np.ndarray, cn: np.ndarray, t_min: float, t_max: float, data_dir: Path, cn_for_snvs:np.ndarray) -> List[int]:
    assert D.shape == cn_for_snvs.shape

    for i in range(0, D.shape[0]):
        for j in range(0, D.shape[1]):
            if cn_for_snvs[i,j] > 0:
                D[i,j] = D[i,j] / cn_for_snvs[i,j]
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

    clustering = [x for _, x in cell_cluster_pairs]
    """
    PLOTTING 
    """
    colors = []
    for i in range(len(cell_cluster_pairs)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    dflt_col = "#808080"  # Unclustered gray
    link_cols = {}

    for i, i12 in enumerate(x[:, :2].astype(int)):
        c1, c2 = (link_cols[y] if y > len(x) else colors[clustering[y]]
                  for y in i12)
        link_cols[i + 1 + len(x)] = c1 if c1 == c2 else dflt_col

    plt.figure(figsize=(50, 50))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    sch.dendrogram(
        x,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        link_color_func=lambda x: link_cols[x],
    )
    plt.savefig(f'{str(data_dir / Path("dendogram"))}.png')

    return [x for _,x in cell_cluster_pairs]





def find_clustering_bottom_up(D: np.ndarray, cn: np.ndarray, t_min: float, t_max: float, data_dir: Path, cn_for_snvs:np.ndarray) -> List[int]:
    min_coverage = t_min
    print(f"Generating clusters from {D.shape[0]} cells with min coverage {min_coverage}")

    x = sch.linkage(cn, method='ward')


    final_clusters = {}

    def coverage(cells: List[int]):
        return np.sum(D[cells, :]) / D.shape[1]

    cluster_to_cell = {cell: [cell] for cell in range(D.shape[0])}
    cluster_id = 0
    while True:
        print(f"Left clusters {len(cluster_to_cell)}")
        to_remove = []
        for cluster, cells in cluster_to_cell.items():
            if coverage(cells) >= min_coverage:
                final_clusters[cluster_id] = cells
                cluster_id += 1
                to_remove.append(cluster)
        for c in to_remove:
            cluster_to_cell.pop(c)

        if len(cluster_to_cell) <= 1:
            break

        coverages = []
        for clust, cells in final_clusters.items():
            coverages.append(coverage(cells))
        for clust, cells in cluster_to_cell.items():
            coverages.append(coverage(cells))

        if sum(coverages) / len(coverages) >= min_coverage:
            break

        clusters = list(set(cluster_to_cell.keys()))
        cn_clust = np.zeros((len(clusters), cn.shape[1]))
        for i, clust in enumerate(clusters):
            cn_clust[i, ] = np.mean(cn[cluster_to_cell[clust]], axis=0)
        clustering = AgglomerativeClustering(n_clusters=len(clusters) - 1, metric='euclidean').fit(cn_clust).labels_
        clustering = list(clustering)
        new_clusters = defaultdict(list)
        assert len(clustering) == len(clusters)
        for i, clust in enumerate(clustering):
            new_clusters[clust].extend(cluster_to_cell[clusters[i]])
        assert len(cluster_to_cell) == len(new_clusters) + 1
        cluster_to_cell = dict(new_clusters)

    for clust, cells in cluster_to_cell.items():
        final_clusters[cluster_id] = cells
        cluster_id += 1

    cell_cluster = []
    for clust, cells in final_clusters.items():
        for c in cells:
            cell_cluster.append((c, clust))
    cell_cluster.sort()
    assert len(cell_cluster) == cn.shape[0]
    clustering = [c[1] for c in cell_cluster]

    """
        PLOTTING 
        """
    colors = []
    for i in range(len(clustering)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    dflt_col = "#808080"  # Unclustered gray
    link_cols = {}

    for i, i12 in enumerate(x[:, :2].astype(int)):
        c1, c2 = (link_cols[y] if y > len(x) else colors[clustering[y]]
                  for y in i12)
        link_cols[i + 1 + len(x)] = c1 if c1 == c2 else dflt_col

    plt.figure(figsize=(50, 50))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    sch.dendrogram(
        x,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8,  # font size for the x axis labels
        link_color_func=lambda x: link_cols[x],
    )
    plt.savefig(f'{str(data_dir / Path("dendogram"))}.png')

    return [c[1] for c in cell_cluster]

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