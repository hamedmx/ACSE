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

                metrics = ACSEMetrics.compute_discrimination_metrics(
                    u_hat_scores,
                    labels
                )

                auarc = ACSEMetrics.compute_auarc(
                    np.array(u_hat_scores),
                    labels
                )

                print(
                    f"  > n={n}: "
                    f"AUROC={metrics['AUROC']:.3f}, "
                    f"AUARC={auarc:.3f}"
                )

                results[n] = {
                    'AUROC': metrics['AUROC'],
                    'AUARC': auarc
                }

            else:
                print(f"  > n={n}: Skipped (Insufficient data)")

        return results

    @staticmethod
    def run_clustering_threshold_ablation(
        responses: List[List[str]],
        labels: np.ndarray
    ) -> Dict[float, float]:
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

            metrics = ACSEMetrics.compute_discrimination_metrics(
                u_hat_scores,
                labels
            )

            auroc = metrics['AUROC']

            print(f"  > Epsilon={epsilon:.2f}: AUROC={auroc:.3f}")

            if epsilon == 0.35:
                print("    (Optimal Threshold)")

            results[epsilon] = auroc

        return results

    @staticmethod
    def compare_base_se_to_acse(
        u_raw: np.ndarray,
        u_inflated: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Comparative Analysis (Base SE vs ACSE).
        """

        print("\n--- [Table 7] Base SE vs. ACSE Comparison ---")

        base_metrics = ACSEMetrics.compute_discrimination_metrics(
            u_raw,
            labels
        )

        acse_metrics = ACSEMetrics.compute_discrimination_metrics(
            u_inflated,
            labels
        )

        deltas = ACSEMetrics.compare_base_vs_acse(
            base_metrics,
            acse_metrics
        )

        print(
            f"  > Base SE: "
            f"AUROC={base_metrics['AUROC']:.3f}, "
            f"FPR@95={base_metrics['FPR@95']:.3f}"
        )

        print(
            f"  > ACSE:    "
            f"AUROC={acse_metrics['AUROC']:.3f}, "
            f"FPR@95={acse_metrics['FPR@95']:.3f}"
        )

        print(
            f"  > Delta:   "
            f"AUROC Gain={deltas['AUROC_Gain']:.3f}, "
            f"FPR95 Reduction={deltas['FPR95_Reduction']:.3f}"
        )

        return deltas

    @staticmethod
    def run_feature_ablation(
        clustering_outputs: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Feature Importance (Leave-One-Out).
        """

        print("\n--- Feature Ablation Analysis ---")

        feature_names = {
            0: "Semantic Entropy (u)",
            1: "Centroid Distance (a_tilde)",
            2: "Cluster Dispersion (d_k)",
            3: "Cluster Size Penalty (g_k)",
            4: "Margin to Threshold (m)"
        }

        # Full model
        engine_full = AdaptiveUncertaintyEngine(
            weights=np.ones(5)
        )

        if engine_full.kappa is None:
            engine_full.kappa = 1.0
            engine_full.tau_ref = 0.5

        scores_full = [
            engine_full.compute_inflated_score(d)['u_hat']
            for d in clustering_outputs
        ]

        metrics_full = ACSEMetrics.compute_discrimination_metrics(
            scores_full,
            labels
        )

        print(f"  > Full Model: AUROC={metrics_full['AUROC']:.3f}")

        results = {}

        # Leave-one-out
        for idx, name in feature_names.items():

            weights = np.ones(5)
            weights[idx] = 0.0

            engine_ablated = AdaptiveUncertaintyEngine(
                weights=weights
            )

            engine_ablated.kappa = engine_full.kappa
            engine_ablated.tau_ref = engine_full.tau_ref

            scores_ablated = [
                engine_ablated.compute_inflated_score(d)['u_hat']
                for d in clustering_outputs
            ]

            metrics_ablated = ACSEMetrics.compute_discrimination_metrics(
                scores_ablated,
                labels
            )

            deltas = ACSEMetrics.compute_feature_ablation_deltas(
                metrics_full,
                metrics_ablated
            )

            drop = deltas['AUROC_Drop']

            print(
                f"  > w/o {name}: "
                f"AUROC={metrics_ablated['AUROC']:.3f} "
                f"(Drop: {drop:.3f})"
            )

            results[name] = drop

        return results

    # ============================================================
    # TABLE 13
    # Weight Sensitivity (Discriminative)
    # ============================================================

    @staticmethod
    def run_weight_sensitivity_ablation(
        clustering_outputs: List[Dict[str, Any]],
        labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:

        print("\n--- Brittleness Weight Sensitivity ---")

        weight_configs = {
            "Uniform": np.array([1/5, 1/5, 1/5, 1/5, 1/5]),

            "Entropy": np.array([
                2/6, 1/6, 1/6, 1/6, 1/6
            ]),

            "Geometry": np.array([
                1/7, 2/7, 2/7, 1/7, 1/7
            ]),

            "Support": np.array([
                1/6, 1/6, 1/6, 2/6, 1/6
            ]),

            "Margin": np.array([
                1/6, 1/6, 1/6, 1/6, 2/6
            ])
        }

        results = {}

        for config_name, weights in weight_configs.items():

            engine = AdaptiveUncertaintyEngine(
                weights=weights
            )

            if engine.kappa is None:
                engine.kappa = 1.0
                engine.tau_ref = 0.5

            scores = []

            for sample in clustering_outputs:

                out = engine.compute_inflated_score(sample)

                scores.append(out['u_hat'])

            metrics = ACSEMetrics.compute_discrimination_metrics(
                scores,
                labels
            )

            auarc = ACSEMetrics.compute_auarc(
                np.array(scores),
                labels
            )

            results[config_name] = {
                "AUROC": metrics["AUROC"],
                "FPR@95": metrics["FPR@95"],
                "FPR@90": metrics["FPR@90"],
                "AUPR": metrics["AUPR"],
                "AUARC": auarc
            }

            print(
                f"  > {config_name}: "
                f"AUROC={metrics['AUROC']:.3f}, "
                f"FPR@95={metrics['FPR@95']:.3f}, "
                f"AUARC={auarc:.3f}"
            )

        return results

    # ============================================================
    # TABLE 14
    # Weight Sensitivity (Conformal)
    # ============================================================

    @staticmethod
    def run_weight_sensitivity_conformal(
        clustering_outputs: List[Dict[str, Any]],
        labels: np.ndarray,
        alpha: float = 0.10
    ) -> Dict[str, Dict[str, float]]:

        print("\n--- Weight Sensitivity Conformal ---")

        weight_configs = {
            "Uniform": np.array([1/5, 1/5, 1/5, 1/5, 1/5]),

            "Entropy": np.array([
                2/6, 1/6, 1/6, 1/6, 1/6
            ]),

            "Geometry": np.array([
                1/7, 2/7, 2/7, 1/7, 1/7
            ]),

            "Support": np.array([
                1/6, 1/6, 1/6, 2/6, 1/6
            ]),

            "Margin": np.array([
                1/6, 1/6, 1/6, 1/6, 2/6
            ])
        }

        results = {}

        for config_name, weights in weight_configs.items():

            engine = AdaptiveUncertaintyEngine(
                weights=weights
            )

            scores = []

            for sample in clustering_outputs:

                out = engine.compute_inflated_score(sample)

                scores.append(out['u_hat'])

            scores = np.array(scores)

            metrics = ACSEMetrics.compute_conformal_metrics(
                uncertainty_scores=scores,
                labels=labels,
                alpha=alpha
            )

            results[config_name] = metrics

            print(
                f"  > {config_name}: "
                f"P-Cov={metrics['P_COV']:.3f}, "
                f"APS={metrics['APS']:.3f}, "
                f"Risk={metrics['RISK']:.3f}"
            )

        return results

    # ============================================================
    # TABLE 15
    # Encoder Sensitivity (Discriminative)
    # ============================================================

    @staticmethod
    def run_encoder_ablation(
        all_responses: List[List[str]],
        labels: np.ndarray,
        encoder_map: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:

        print("\n--- Encoder Sensitivity ---")

        results = {}

        for encoder_name, model_name in encoder_map.items():

            print(f"\n  > Evaluating encoder: {encoder_name}")

            clusterer = SemanticClusterer(
                encoder_model=model_name
            )

            engine = AdaptiveUncertaintyEngine()

            scores = []

            for responses in all_responses:

                cluster_data = clusterer.process_prompt(
                    responses
                )

                out = engine.compute_inflated_score(
                    cluster_data
                )

                scores.append(out['u_hat'])

            metrics = ACSEMetrics.compute_discrimination_metrics(
                scores,
                labels
            )

            auarc = ACSEMetrics.compute_auarc(
                np.array(scores),
                labels
            )

            results[encoder_name] = {
                "AUROC": metrics["AUROC"],
                "FPR@95": metrics["FPR@95"],
                "FPR@90": metrics["FPR@90"],
                "AUPR": metrics["AUPR"],
                "AUARC": auarc
            }

            print(
                f"    AUROC={metrics['AUROC']:.3f}, "
                f"FPR@95={metrics['FPR@95']:.3f}, "
                f"AUARC={auarc:.3f}"
            )

        return results

    # ============================================================
    # TABLE 16
    # Encoder Sensitivity (Conformal)
    # ============================================================

    @staticmethod
    def run_encoder_conformal_ablation(
        all_responses: List[List[str]],
        labels: np.ndarray,
        encoder_map: Dict[str, str],
        alpha: float = 0.10
    ) -> Dict[str, Dict[str, float]]:

        print("\n--- Encoder Conformal Sensitivity ---")

        results = {}

        for encoder_name, model_name in encoder_map.items():

            clusterer = SemanticClusterer(
                encoder_model=model_name
            )

            engine = AdaptiveUncertaintyEngine()

            scores = []

            for responses in all_responses:

                cluster_data = clusterer.process_prompt(
                    responses
                )

                out = engine.compute_inflated_score(
                    cluster_data
                )

                scores.append(out['u_hat'])

            scores = np.array(scores)

            metrics = ACSEMetrics.compute_conformal_metrics(
                uncertainty_scores=scores,
                labels=labels,
                alpha=alpha
            )

            results[encoder_name] = metrics

            print(
                f"  > {encoder_name}: "
                f"P-Cov={metrics['P_COV']:.3f}, "
                f"APS={metrics['APS']:.3f}, "
                f"Risk={metrics['RISK']:.3f}"
            )

        return results
