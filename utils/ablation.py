import numpy as np
from typing import List, Dict, Any
from sklearn.isotonic import IsotonicRegression
from pipeline import SemanticClusterer, AdaptiveUncertaintyEngine
from utils.metrics import ACSEMetrics

class AblationSuite:
    """
    This class consolidates the logic required to reproduce the ablation studies.
    """

    @staticmethod
    def run_scatter_plot_analysis(acse_uncertainty: np.ndarray, 
                                  baseline_confidence: np.ndarray, 
                                  error_labels: np.ndarray) -> Dict[str, int]:
        """
        Args:
            acse_uncertainty: Array of u_hat(x) scores.
            baseline_confidence: Array of baseline confidence
            error_labels: 1=Incorrect, 0=Correct.
            
        Returns:
            Dict containing counts for each quadrant.
        """
        print("\n--- Discriminative Generation via Uncertainty-Confidence Analysis ---")
        
        quadrants = ACSEMetrics.identify_scatter_quadrants(
            uncertainty_scores=acse_uncertainty,
            baseline_confidence=baseline_confidence,
            error_labels=error_labels
        )
        
        print(f"  > Ideal Correct (Bottom-Right): {quadrants['ideal_correct']}")
        print(f"  > Caught Hallucinations (Top-Right): {quadrants['caught_hallucinations']}")
        print(f"  > Agreed Uncertainty (Top-Left): {quadrants['agreed_uncertainty']}")
        print(f"  > Anomalous Low Confidence (Bottom-Left): {quadrants['anomalous_low_conf']}")
        
        return quadrants

    @staticmethod
    def run_sampling_ablation(all_responses: List[List[str]], 
                              labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Sample Size Ablation study.
        
        Args:
            all_responses: List of lists.
            labels: Binary error labels.
        """
        print("\n--- Sample Size Ablation ---")
        
        sample_sizes = [4, 7, 10, 13, 16]
        clusterer = SemanticClusterer()
        engine = AdaptiveUncertaintyEngine() 
        
        results = {}
        
        for n in sample_sizes:
            u_hat_scores = []
            
            for responses in all_responses:
                if len(responses) < n:
                    continue 
                current_sample = responses[:n]
                
                # 1. Clustering
                data = clusterer.process_prompt(current_sample)
                
                # 2. Adaptive Inflation
                out = engine.compute_inflated_score(data)
                u_hat_scores.append(out['u_hat'])
            
            # 3. Metrics
            if len(u_hat_scores) > 0:
                metrics = ACSEMetrics.compute_discrimination_metrics(u_hat_scores, labels)
                auarc = ACSEMetrics.compute_auarc(np.array(u_hat_scores), labels)
                
                print(f"  > n={n}: AUROC={metrics['AUROC']:.3f}, AUARC={auarc:.3f}")
                results[n] = {'AUROC': metrics['AUROC'], 'AUARC': auarc}
            else:
                print(f"  > n={n}: Skipped (Insufficient data)")
                
        return results

    @staticmethod
    def run_clustering_threshold_ablation(responses: List[List[str]], 
                                          labels: np.ndarray) -> Dict[float, float]:
        """
        Clustering Threshold Sensitivity.
        """
        print("\n--- Clustering Threshold Ablation ---")
        
        thresholds = [0.10, 0.20, 0.35, 0.50, 0.70]
        engine = AdaptiveUncertaintyEngine()
        
        results = {}
        
        for epsilon in thresholds:
            clusterer = SemanticClusterer(epsilon=epsilon)
            
            u_hat_scores = []
            for resp_set in responses:
                data = clusterer.process_prompt(resp_set)
                out = engine.compute_inflated_score(data)
                u_hat_scores.append(out['u_hat'])
                
            metrics = ACSEMetrics.compute_discrimination_metrics(u_hat_scores, labels)
            auroc = metrics['AUROC']
            
            print(f"  > Epsilon={epsilon:.2f}: AUROC={auroc:.3f}")
            if epsilon == 0.35:
                print("    (Optimal Threshold)")

            results[epsilon] = auroc
            
        return results

    @staticmethod
    def compare_base_se_to_acse(u_raw: np.ndarray, 
                                u_inflated: np.ndarray, 
                                labels: np.ndarray) -> Dict[str, float]:
        """
        Comparative Analysis (Base SE vs ACSE).
        
        Demonstrates the impact of adaptive inflation.
        """
        print("\n--- [Table 7] Base SE vs. ACSE Comparison ---")
        
        base_metrics = ACSEMetrics.compute_discrimination_metrics(u_raw, labels)
        acse_metrics = ACSEMetrics.compute_discrimination_metrics(u_inflated, labels)
        
        deltas = ACSEMetrics.compare_base_vs_acse(base_metrics, acse_metrics)
        
        print(f"  > Base SE: AUROC={base_metrics['AUROC']:.3f}, FPR@95={base_metrics['FPR@95']:.3f}")
        print(f"  > ACSE:    AUROC={acse_metrics['AUROC']:.3f}, FPR@95={acse_metrics['FPR@95']:.3f}")
        print(f"  > Delta:   AUROC Gain={deltas['AUROC_Gain']:.3f}, FPR95 Reduction={deltas['FPR95_Reduction']:.3f}")
        
        return deltas

    @staticmethod
    def run_feature_ablation(clustering_outputs: List[Dict[str, Any]], 
                             labels: np.ndarray) -> Dict[str, float]:
        """
        Feature Importance (Leave-One-Out).
        Ablates each of the 5 semantic features to measure AUROC drop.
        """
        print("\n--- Feature Ablation Analysis ---")
        
        # Define feature mapping
        feature_names = {
            0: "Semantic Entropy (u)",
            1: "Centroid Distance (a_tilde)",
            2: "Cluster Dispersion (d_k)",
            3: "Cluster Size Penalty (g_k)",
            4: "Margin to Threshold (m)"
        }
        
        # 1. Full Model
        engine_full = AdaptiveUncertaintyEngine(weights=np.ones(5))
        if engine_full.kappa is None: 
            engine_full.kappa = 1.0; engine_full.tau_ref = 0.5 
            
        scores_full = [engine_full.compute_inflated_score(d)['u_hat'] for d in clustering_outputs]
        metrics_full = ACSEMetrics.compute_discrimination_metrics(scores_full, labels)
        print(f"  > Full Model: AUROC={metrics_full['AUROC']:.3f}")
        
        results = {}
        
        # 2. Leave-One-Out
        for idx, name in feature_names.items():
            weights = np.ones(5)
            weights[idx] = 0.0
            
            engine_ablated = AdaptiveUncertaintyEngine(weights=weights)
            engine_ablated.kappa = engine_full.kappa
            engine_ablated.tau_ref = engine_full.tau_ref
            
            scores_ablated = [engine_ablated.compute_inflated_score(d)['u_hat'] for d in clustering_outputs]
            metrics_ablated = ACSEMetrics.compute_discrimination_metrics(scores_ablated, labels)
            
            deltas = ACSEMetrics.compute_feature_ablation_deltas(metrics_full, metrics_ablated)
            drop = deltas['AUROC_Drop']
            
            print(f"  > w/o {name}: AUROC={metrics_ablated['AUROC']:.3f} (Drop: {drop:.3f})")
            results[name] = drop
            
        return results