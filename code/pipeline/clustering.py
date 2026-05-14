import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

class SemanticClusterer:
    """
    Implements the Semantic Clustering and Entropy pipeline.
    
    This class transforms raw response strings into a semantic distribution to calculate
    the Base Uncertainty Score u(x).
    
    Attributes:
        encoder (SentenceTransformer): The model f(.) used to map responses to R^d.
        epsilon (float): The clustering threshold epsilon used to cut the HAC dendrogram.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', epsilon: float = 0.35):
        self.encoder = SentenceTransformer(model_name)
        self.epsilon = epsilon

    def embed_responses(self, responses: List[str]) -> np.ndarray:
        """
        Maps the response set Y(x) to the embedding set E.
        
        Args:
            responses: The list of n generated response strings {y_1, ..., y_n}.
            
        Returns:
            np.ndarray: Matrix of embeddings where each row v_i has unit L2 norm.
        """
        embeddings = self.encoder.encode(responses)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-9)
        
        return normalized_embeddings

    def cluster_responses(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Performs Hierarchical Agglomerative Clustering (HAC) on embeddings E.
        
        Args:
            embeddings: Unit-norm embedding matrix (n x d).
            
        Returns:
            Tuple: 
                - cluster_labels: Array of size n assigning each v_i to a cluster number.
                - K: The total number of semantic clusters found.
        """
        n = len(embeddings)
        
        if n < 2:
            return np.ones(n, dtype=int), 1

        distances = pdist(embeddings, metric='cosine')
        Z = linkage(distances, method='average')
        cluster_labels = fcluster(Z, t=self.epsilon, criterion='distance')
        K = np.max(cluster_labels)
        
        return cluster_labels, K

    def compute_centroids(self, embeddings: np.ndarray, cluster_labels: np.ndarray, K: int) -> np.ndarray:
        """
        Computes the unit-norm centroid c_k for each cluster C_k.
        
        Args:
            embeddings: Response embeddings v_i.
            cluster_labels: Hard assignments from HAC.
            K: Number of clusters.
            
        Returns:
            np.ndarray: Matrix of centroids (K x d).
        """
        centroids = []

        for k in range(1, K + 1):
            mask = (cluster_labels == k)
            cluster_vectors = embeddings[mask]
            mean_vec = np.mean(cluster_vectors, axis=0)
            norm = np.linalg.norm(mean_vec)
            centroid = mean_vec / (norm + 1e-9)
            centroids.append(centroid)
            
        return np.array(centroids)

    def compute_soft_assignments(self, embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Computes the soft assignment matrix S where s_ik is the membership of y_i in C_k.
        
        Args:
            embeddings: Response embeddings v_i (n x d).
            centroids: Cluster centroids c_k (K x d).
            
        Returns:
            np.ndarray: Soft assignment matrix s (n x K).
        """
        cosine_sims = np.dot(embeddings, centroids.T) 
        a_ik = (1.0 + cosine_sims) / 2.0
        row_sums = np.sum(a_ik, axis=1, keepdims=True)
        s_ik = a_ik / (row_sums + 1e-9)
        
        return s_ik

    def compute_entropy(self, s_ik: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Computes the Base Semantic Entropy u(x) from soft assignments.
        
        Args:
            s_ik: Soft assignment matrix (n x K).
            
        Returns:
            Tuple:
                - u_x: Normalized semantic entropy (scalar).
                - P_Ck: The distribution over semantic clusters (vector of size K).
        """
        n, K = s_ik.shape

        P_Ck = np.mean(s_ik, axis=0)
        H_sem = -np.sum(P_Ck * np.log(P_Ck + 1e-9))
        if K <= 1:
            u_x = 0.0
        else:
            u_x = H_sem / np.log(K)
            
        return u_x, P_Ck

    def process_prompt(self, responses: List[str]) -> Dict:
        """
        Orchestrates the full clustering pipeline for a single prompt x.
        
        Args:
            responses: List of n generated response strings.
            
        Returns:
            Dict containing:
                - 'u_x': Base semantic entropy.
                - 'embeddings': The n x d embedding matrix.
                - 'centroids': The K x d centroid matrix.
                - 'probs': The distribution P(C_k).
                - 'cluster_ids': Hard cluster assignments.
                - 'soft_assignments': The n x K matrix S.
        """
        # Step 1: Embed responses 
        embeddings = self.embed_responses(responses)
        
        # Step 2: HAC Clustering 
        cluster_labels, K = self.cluster_responses(embeddings)
        
        # Step 3: Compute Centroids 
        centroids = self.compute_centroids(embeddings, cluster_labels, K)
        
        # Step 4: Soft Assignments
        s_ik = self.compute_soft_assignments(embeddings, centroids)
        
        # Step 5: Semantic Entropy 
        u_x, P_Ck = self.compute_entropy(s_ik)
        
        return {
            'u_x': u_x,
            'embeddings': embeddings,
            'centroids': centroids,
            'probs': P_Ck,           
            'cluster_ids': cluster_labels,
            'soft_assignments': s_ik 
        }