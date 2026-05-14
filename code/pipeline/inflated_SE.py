import numpy as np
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Any

class AdaptiveUncertaintyEngine:
    """
    Implements the Adaptive Inflation Mechanism.
    
    This class is responsible for computing the Inflated Semantic Uncertainty score u_hat(x).
    It inflates the base entropy u(x) based on the structural brittleness of the 
    semantic clusters, penalized by five robust geometric features.
    
    The inflation logic operates strictly at the prompt level by aggregating 
    distributional properties of the response set Y(x).
    
    Attributes:
        kappa (float): Dataset-specific scaling constant for cluster size.
        tau_ref (float): Reference confidence threshold from calibration set.
        weights (np.ndarray): Weight vector w for risk aggregation.
    """

    def __init__(self, 
                 weights: np.ndarray = np.ones(5), 
                 gamma: float = 0.75):
        """
        Initialize the engine with weighting and calibration parameters.
        
        Args:
            weights: The non-negative weight vector w encoding feature contribution.
            gamma: The quantile level for determining tau_ref. 
        """
        self.weights = weights
        self.gamma = gamma
        self.kappa = None 
        self.tau_ref = None

    def fit_calibration_stats(self, calibration_results: List[Dict[str, Any]]):
        """
        Computes the frozen distributional constants kappa and tau_ref from D_cal.
        
        These constants are derived once from the unlabeled calibration set 
        and frozen during inference to preserve conformal validity.
        
        Args:
            calibration_results: List of outputs from SemanticClusterer.process_prompt() 
                                 for all x in D_cal.
        """
        # 1. Compute kappa (Scaling constant for cluster sparsity)
        max_cluster_sizes = []
        
        for res in calibration_results:
            labels = res['cluster_ids']
            unique, counts = np.unique(labels, return_counts=True)
            max_size = np.max(counts)
            max_cluster_sizes.append(max_size)

        self.kappa = np.median(max_cluster_sizes)
        
        # 2. Compute tau_ref (Reference Threshold for Overconfidence)
        u_scores = [res['u_x'] for res in calibration_results]
        self.tau_ref = np.quantile(u_scores, self.gamma)

        if self.tau_ref < 1e-6:
            self.tau_ref = 1e-6

    def compute_features(self, clustering_data: Dict[str, Any]) -> np.ndarray:
        """
        Synthesizes the five normalized prompt-level robustness features F.
        
        Args:
            clustering_data: Output dictionary from SemanticClusterer containing:
                             u_x, embeddings, centroids, probs, cluster_ids, soft_assignments.
                             
        Returns:
            np.ndarray: A vector of 5 features [u, a_tilde, d_k, g_k, m].
        """
        if self.kappa is None or self.tau_ref is None:
            raise ValueError("Calibration statistics (kappa, tau_ref) must be fitted first.")

        u_x = clustering_data['u_x']
        embeddings = clustering_data['embeddings'] 
        centroids = clustering_data['centroids']   
        probs = clustering_data['probs']           
        soft_assigns = clustering_data['soft_assignments'] 
        cluster_ids = clustering_data['cluster_ids']
        
        # Identify the dominant cluster k* 
        k_star_idx = np.argmax(probs) 
        
        # Corresponding centroid c_k*
        c_star = centroids[k_star_idx]
        
        # Identify the cluster label corresponding to k* (assuming 1-based indexing from HAC).
        k_star_label = k_star_idx + 1
        
        # --- Feature 1: Semantic Entropy u(x) ---
        f1_u = u_x
        
        # --- Feature 2: Centroid Distance a_tilde(x) ---
        i_star_idx = np.argmax(soft_assigns[:, k_star_idx])
        v_i_star = embeddings[i_star_idx]
        cos_sim = 1.0 - cosine(v_i_star, c_star)
        a_i_star = (1.0 + cos_sim) / 2.0
        f2_a_tilde = 1.0 - a_i_star
        
        # --- Feature 3: Dominant Cluster Dispersion d_k*(x) ---
        mask_dom = (cluster_ids == k_star_label)
        dom_embeddings = embeddings[mask_dom]
        
        if len(dom_embeddings) > 0:
            dissimilarities = [cosine(v, c_star) for v in dom_embeddings]
            sum_dissim = np.sum(dissimilarities)
            size_dom = len(dom_embeddings)
            f3_dispersion = (1.0 / (2.0 * size_dom)) * sum_dissim
        else:
            f3_dispersion = 0.0
            
        # --- Feature 4: Dominant Cluster Sparsity g_k*(x) ---
        size_dom = np.sum(mask_dom)
        if size_dom > 0:
            ratio = self.kappa / size_dom
            f4_sparsity = min(1.0, ratio)
        else:
            f4_sparsity = 1.0
            
        # --- Feature 5: Margin to Threshold m(x) ---
        ratio_m = u_x / self.tau_ref
        f5_margin = max(0.0, 1.0 - ratio_m)
        
        return np.array([f1_u, f2_a_tilde, f3_dispersion, f4_sparsity, f5_margin])

    def compute_inflated_score(self, clustering_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes lambda(x) and the final inflated uncertainty u_hat(x).
        
        Args:
            clustering_data: Output from SemanticClusterer.
            
        Returns:
            Dict containing:
                - 'u_hat': The inflated semantic uncertainty.
                - 'lambda': The inflation factor.
                - 'R_x': The composite risk score.
                - 'features': The vector of 5 semantic features.
        """
        # 1. Compute the 5 features
        features = self.compute_features(clustering_data)
        
        # 2. Compute Composite Risk R(x) 
        weighted_sum = np.dot(self.weights, features)
        sum_weights = np.sum(self.weights)
        R_x = weighted_sum / (sum_weights + 1e-9)
        R_x = np.clip(R_x, 0.0, 1.0)
        
        # 3. Compute Inflation Factor lambda(x) 
        lambda_x = 2.0 / (2.0 - R_x)
        
        # 4. Compute Inflated Uncertainty u_hat(x) 
        u_x = clustering_data['u_x']
        
        numerator = lambda_x * u_x
        denominator = 1.0 + (lambda_x - 1.0) * u_x
        if denominator < 1e-9:
            u_hat = 0.0
        else:
            u_hat = numerator / denominator
            
        u_hat = np.clip(u_hat, 0.0, 1.0)
        
        return {
            'u_hat': u_hat,
            'lambda': lambda_x,
            'R_x': R_x,
            'features': features
        }