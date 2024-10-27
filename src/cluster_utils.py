import umap
import random
import tiktoken
import numpy as np
from abc import ABC
from abc import abstractmethod
from typing import List,Optional
from sklearn.mixture import GaussianMixture
from tree_structures import Node
from logger import get_logger

# set a random for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)

logger = get_logger(__name__)

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings) # 142 * 10
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings)) #每个聚类数量对应的贝叶斯信息准则（BIC）值
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings) # 13个类别
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings) # 142 * 13
    labels = [np.where(prob > threshold)[0] for prob in probs] # 142 * 1，每个样本属于那个类别
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    ) # 236 * 1， 10
    logger.info(f"*********该层一共{n_global_clusters}个父特征*************")
    if verbose:
        logger.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))] # 142 * 0
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[ # 142* 1526, #13*1536
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logger.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings( # 13 * 10
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logger.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logger.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self,embeddings,**kwargs) -> List[List[int]]:
        pass

class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
            nodes:List[Node],
            embedding_model_name:str,
            max_length_in_cluster: int = 3500,
            tokenizer = tiktoken.get_encoding("cl100k_base"),
            reduction_dimension: int = 10,
            threshold: float = 0.1,
            verbose: bool = False,
    ) -> List[List[Node]]:
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes]) # 142 * 1636
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )
        node_clusters = []
        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]
            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue
            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )
            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logger.info(f"reclustering cluster with {len(cluster_nodes)} nodes")
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(cluster_nodes, embedding_model_name, max_length_in_cluster)
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
