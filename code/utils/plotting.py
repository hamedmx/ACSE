import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple

class ACSEPlotter:
    """
    Visualization engine for the ACSE framework.
    """

    def __init__(self):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

        self.colors = {
            "ACSE (Ours)": "tab:blue", 
            "CAP": "purple",           
            "DDCRP-CP": "green",       
            "SU": "gray",              

            "Correct": "#1f77b4",     
            "Incorrect": "#d62728"  
        }
        
        self.markers = {
            "ACSE (Ours)": "o",       
            "CAP": "^",              
            "DDCRP-CP": "s",         
            "SU": "x"                 
        }

    def plot_sensitivity_analysis(self, 
                                  results: Dict[str, Dict[str, List[float]]], 
                                  alphas: List[float],
                                  save_prefix: str = "sensitivity"):
        """
        Miscoverage Sensitivity Analysis.
        """
        metrics_config = [
            ("Coverage", "Empirical Coverage", "(a) Empirical Coverage", "coverage"),
            ("SSCV", "SSCV", "(b) SSCV", "sscv"),
            ("Selective_Risk", "Guaranteed Risk", "(c) Selective Risk", "risk"),
            ("Acceptance_Rate", "Acceptance Rate (%)", "(d) Acceptance Rate", "acceptance")
        ]
        
        for metric_key, y_label, title, suffix in metrics_config:
            plt.figure(figsize=(7, 6))
        
            if metric_key == "Coverage":
                plt.plot(alphas, [1 - a for a in alphas], 
                         color="black", linestyle="--", label="Target (1-α)", linewidth=2, alpha=0.7)
            elif metric_key == "Selective_Risk":
                plt.plot(alphas, alphas, 
                         color="black", linestyle="--", label="Target (α)", linewidth=2, alpha=0.7)

            for method, data in results.items():
                if metric_key in data:
                    values = data[metric_key]
                    
                    if metric_key == "Acceptance_Rate" and max(values) <= 1.0:
                        values = [v * 100 for v in values]
                        
                    plt.plot(alphas, values, 
                             label=method,
                             color=self.colors.get(method, "black"),
                             marker=self.markers.get(method, "."),
                             markersize=8,
                             linewidth=2.5)

            # 3. Formatting
            plt.xlabel(r"Miscoverage Level ($\alpha$)", fontsize=24)
            plt.ylabel(y_label, fontsize=24)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.xlim(min(alphas), max(alphas))
            
            # Legend positioning
            plt.legend(loc='best', frameon=True, fontsize=16)
            
            # 4. Save
            filename = f"{save_prefix}_{suffix}.pdf"
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved: {filename}")

    def plot_scatter_analysis(self, 
                              acse_scores: np.ndarray, 
                              baseline_scores: np.ndarray, 
                              labels: np.ndarray,
                              baseline_name: str,
                              save_path: str = "scatter_plot.pdf"):
        """
        Visualises Discriminative Generation based on Uncertainty-Confidence Analysis.
        """
        plt.figure(figsize=(7, 7))
        correct_mask = (labels == 0)
        incorrect_mask = (labels == 1)
        plt.scatter(baseline_scores[correct_mask], acse_scores[correct_mask],
                    c=self.colors["Correct"], label="Correct", alpha=0.6, s=20, marker='o', edgecolors='none')
        
        plt.scatter(baseline_scores[incorrect_mask], acse_scores[incorrect_mask],
                    c=self.colors["Incorrect"], label="Incorrect", alpha=0.6, s=20, marker='o', edgecolors='none')

        plt.axhline(y=0.5, color='black', linestyle=':', linewidth=2.5, alpha=0.7)
        plt.axvline(x=0.5, color='black', linestyle=':', linewidth=2.5, alpha=0.7)
        
        plt.xlabel(f"{baseline_name} Confidence", fontsize=24)
        plt.ylabel(r"ACSE Uncertainty ($\hat{u}$)", fontsize=24)
        
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        
        # Legend (Top Left usually)
        plt.legend(loc="upper left", frameon=True, fontsize=16, markerscale=1.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_sample_size_sensitivity(self, 
                                     sample_sizes: List[int],
                                     auroc_scores: List[float],
                                     auarc_scores: List[float],
                                     save_prefix: str = "sample_size"):
        """
        Sample Size Ablation
        """
        # --- Plot 1: AUROC vs n ---
        plt.figure(figsize=(7, 5))
        plt.plot(sample_sizes, auroc_scores, 'o-', color='#003366', linewidth=2.5, markersize=9, label="AUROC")
        plt.axvline(x=10, color='#999999', linestyle='--', linewidth=2, label="Optimal (n=10)")
        
        plt.xlabel(r"Number of Response Samples ($n$)", fontsize=24)
        plt.ylabel("AUROC Score", fontsize=24)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_auroc.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_prefix}_auroc.pdf")
        
        # --- Plot 2: AUARC vs n ---
        plt.figure(figsize=(7, 5))
        plt.plot(sample_sizes, auarc_scores, 'o-', color='#8B0000', linewidth=2.5, markersize=9, label="AUARC")
        plt.axvline(x=10, color='#999999', linestyle='--', linewidth=2, label="Optimal (n=10)")
        
        plt.xlabel(r"Number of Response Samples ($n$)", fontsize=24)
        plt.ylabel("AUARC Score", fontsize=24)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_auarc.pdf", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {save_prefix}_auarc.pdf")

    def plot_threshold_sensitivity(self, 
                                   thresholds: List[float],
                                   auroc_scores: List[float],
                                   save_path: str = "threshold_ablation.pdf"):
        """
        Sensitivity analysis of epsilon.
        """
        plt.figure(figsize=(8, 5.5))
        
        # Main Curve
        plt.plot(thresholds, auroc_scores, 'o-', linewidth=2.5, color='#1f77b4', markersize=8, label="AUROC")
        
        # Highlight Optimal Point
        opt_idx = np.argmax(auroc_scores)
        opt_eps = thresholds[opt_idx]
        opt_score = auroc_scores[opt_idx]
        
        plt.plot(opt_eps, opt_score, 'o', color='red', markersize=12, zorder=5)
        plt.axvline(x=opt_eps, color='red', linestyle='--', alpha=0.6, label=f"Optimal ({opt_eps})")
        
        # Over-Fragmentation
        plt.annotate("Over-Fragmentation", 
                     xy=(0.15, auroc_scores[0]), 
                     xytext=(0.15, auroc_scores[0] - 0.04),
                     fontsize=12, fontweight='bold',
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='center')
        
        # Under-Fragmentation
        plt.annotate("Under-Fragmentation", 
                     xy=(0.6, auroc_scores[-1] + 0.05), 
                     xytext=(0.6, auroc_scores[-1] + 0.1),
                     fontsize=12, fontweight='bold',
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='center')

        plt.xlabel(r"Clustering Threshold ($\epsilon$)", fontsize=24)
        plt.ylabel("AUROC", fontsize=24)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc='lower left', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    def plot_feature_ablation(self, 
                              feature_names: List[str],
                              auroc_drops: List[float],
                              full_model_scores: List[float],
                              save_path: str = "feature_ablation.pdf"):
        """
        Horizontal bar chart for feature importance.
        """
        plt.figure(figsize=(9, 5))
        
        # Normalize for color intensity
        norm_drops = np.array(auroc_drops) / (max(auroc_drops) + 1e-9)
        colors = [plt.cm.Reds(0.4 + 0.5 * d) for d in norm_drops]
        
        # Bar Chart
        bars = plt.barh(feature_names, auroc_drops, color=colors, height=0.65)
        
        # Annotations
        for bar, drop, absolute in zip(bars, auroc_drops, full_model_scores):
            width = bar.get_width()
            label_text = f"-{drop:.3f}\n(Score: {absolute:.3f})"
            plt.text(width + 0.002, 
                     bar.get_y() + bar.get_height()/2, 
                     label_text, 
                     ha='left', 
                     va='center', 
                     fontsize=13, 
                     color='black')
            
        plt.xlabel("Decrease in AUROC", fontsize=24)
        plt.tick_params(axis='y', labelsize=24)
        
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.gca().invert_yaxis()
        sns.despine()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")