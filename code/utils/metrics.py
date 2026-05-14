import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, auc
from typing import Dict, List, Tuple, Any, Optional

class ACSEMetrics:
    """
    Implements the evaluation protocols for the ACSE framework.
    
    This utility handles the computation of:
    1. Discriminative Performance (AUROC, AUPR, FPR@95) for hallucination detection.
    2. Selective Generation Performance (AUARC).
    3. Conformal Validity (Coverage, Risk, SSCV) ensuring statistical guarantees.
    4. Probabilistic Calibration (ECE, Brier Score) for uncertainty reliability.
    """

    @staticmethod
    def compute_discrimination_metrics(uncertainty_scores: np.ndarray, 
                                       error_labels: np.ndarray) -> Dict[str, float]:
        """
        Computes discrimination metrics to evaluate Hallucination Detection.
        
        Args:
            uncertainty_scores (np.ndarray): The computed uncertainty u_hat(x).
            error_labels (np.ndarray): Binary labels E(x) where 1=Incorrect/Hallucination, 0=Correct.
            
        Returns:
            Dict: {
                'AUROC': Area Under ROC curve.
                'AUPR': Area Under Precision-Recall curve.
                'FPR@95': False Positive Rate at 95% Recall (TPR).
            }
        """
        scores = np.array(uncertainty_scores)
        labels = np.array(error_labels)
        
        # 1. AUROC (Area Under ROC Curve)
        try:
            auroc = roc_auc_score(labels, scores)
        except ValueError:
            auroc = 0.0
        
        # 2. AUPR (Area Under Precision-Recall Curve)
        try:
            aupr = average_precision_score(labels, scores)
        except ValueError:
            aupr = 0.0
        
        # 3. FPR@95 (False Positive Rate at 95% Recall)
        if len(np.unique(labels)) < 2:
            return {"AUROC": 0.0, "AUPR": 0.0, "FPR@95": 0.0}
        desc_indices = np.argsort(scores)[::-1]
        sorted_labels = labels[desc_indices]
        
        P = np.sum(labels == 1) 
        N = np.sum(labels == 0) 
        
        if P == 0:
            fpr_95 = 0.0
        else:
            tpr_cumsum = np.cumsum(sorted_labels) / P
            idx_95 = np.searchsorted(tpr_cumsum, 0.95)
            if idx_95 < len(sorted_labels):
                fp_at_95 = np.sum(1 - sorted_labels[:idx_95+1])
                fpr_95 = fp_at_95 / N if N > 0 else 0.0
            else:
                fpr_95 = 1.0
            
        return {
            "AUROC": auroc,
            "AUPR": aupr,
            "FPR@95": fpr_95
        }

    @staticmethod
    def compute_auarc(uncertainty_scores: np.ndarray, 
                      error_labels: np.ndarray) -> float:
        """
        Computes Area Under Accuracy-Rejection Curve (AUARC). 
        Args:
            uncertainty_scores: Higher score = more likely to reject.

        Returns:
            float: AUARC score.
        """
        sorted_indices = np.argsort(uncertainty_scores)[::-1]
        sorted_errors = np.array(error_labels)[sorted_indices]
        
        n = len(sorted_errors)
        accuracies = []
        rejection_rates = []

        for i in range(n + 1):
            rejection_rate = i / n

            if i < n:
                accepted_slice = sorted_errors[i:]
                n_accepted = len(accepted_slice)
                n_correct = n_accepted - np.sum(accepted_slice)
                accuracy = n_correct / n_accepted
            else:
                accuracy = 1.0
                
            accuracies.append(accuracy)
            rejection_rates.append(rejection_rate)

        return auc(rejection_rates, accuracies)

    @staticmethod
    def compute_conformal_metrics(prediction_sets: List[List[str]],
                                  correct_responses: List[str],
                                  acceptance_decisions: List[bool],
                                  prompt_errors: List[int],
                                  alpha: float) -> Dict[str, float]:
        """
        Computes Conformal Prediction metrics.
        
        Args:
            prediction_sets: List of response sets C_alpha(x).
            correct_responses: Ground truth strings (for coverage check).
            acceptance_decisions: List of booleans (Accepted/Abstained).
            prompt_errors: Binary labels E(x) (1=Incorrect, 0=Correct).
            alpha: Target miscoverage level.
            
        Returns:
            Dict containing Coverage, APS, SSCV, Selective Risk, Acceptance Rate.
        """
        decisions = np.array(acceptance_decisions)
        errors = np.array(prompt_errors)
        
        # --- 1. Acceptance Rate ---
        acc_rate = np.mean(decisions)
        
        # --- 2. Selective Risk ---
        if np.sum(decisions) > 0:
            selective_risk = np.mean(errors[decisions])
        else:
            selective_risk = 0.0
            
        # --- 3. Conformal Truth Coverage (CTC) ---
        correct_mask = (errors == 0) 
        if np.sum(correct_mask) > 0:
            # CTC measures how often the model accepts a prompt that it got right.
            ctc = np.sum(decisions[correct_mask]) / np.sum(correct_mask)
        else:
            ctc = 0.0
            
        # --- 4. Empirical Coverage ---
        correct_indices = [i for i, e in enumerate(prompt_errors) if e == 0]
        N_0 = len(correct_indices)
        
        covered_list = []
        set_sizes_correct = [] 
        
        for i in correct_indices:
            p_set = prediction_sets[i]
            truth = correct_responses[i]

            is_covered = 1.0 if (truth in p_set) else 0.0
            covered_list.append(is_covered)
            set_sizes_correct.append(len(p_set))
            
        if N_0 > 0:
            emp_coverage = np.mean(covered_list)
        else:
            emp_coverage = 0.0
            
        # --- 5. Average Prediction Set Size (APS) ---
        if len(set_sizes_correct) > 0:
            aps = np.mean(set_sizes_correct)
        else:
            aps = 0.0
        
        # --- 5. Size-Stratified Coverage Violation (SSCV) ---
        
        arr_sizes = np.array(set_sizes_correct)
        arr_covered = np.array(covered_list)
        
        max_violation = 0.0
        target_cov = 1.0 - alpha

        bins = [
            (1, 2),  
            (3, 5),  
            (6, 7),  
            (8, 10)  
        ]
        
        if len(arr_sizes) > 0:
            for (low, high) in bins:
                mask = (arr_sizes >= low) & (arr_sizes <= high)
                
                if np.sum(mask) > 0:
                    cov_b = np.mean(arr_covered[mask])
                    violation = max(0.0, target_cov - cov_b)
                    
                    if violation > max_violation:
                        max_violation = violation
                        
            sscv = max_violation
        else:
            sscv = 0.0

        return {
            "CTC": ctc,
            "Coverage": emp_coverage,
            "APS": aps,
            "SSCV": sscv,
            "Selective_Risk": selective_risk,
            "Acceptance_Rate": acc_rate
        }

    @staticmethod
    def compute_calibration_metrics(probs_correct: np.ndarray, 
                                    labels: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, float]:
        """
        Computes Probabilistic Calibration metrics.
        
        Args:
            probs_correct: Confidence scores (1 - u_hat).
            labels: Binary correctness (1=Correct, 0=Incorrect).
            
        Returns:
            Dict: {'ECE': val, 'Brier': val}
        """
        # Brier Score
        brier = brier_score_loss(labels, probs_correct)
        
        # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            
            in_bin = (probs_correct > bin_lower) & (probs_correct <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_conf = np.mean(probs_correct[in_bin])
                avg_acc = np.mean(labels[in_bin])
                ece += np.abs(avg_conf - avg_acc) * prop_in_bin
                
        return {"ECE": ece, "Brier": brier}

  